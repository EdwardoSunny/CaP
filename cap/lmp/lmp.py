import os
import numpy as np
import threading
import copy
from openai import OpenAI

# imports for LMPs
import shapely
import ast
import astunparse
from time import sleep
from shapely.geometry import *
from shapely.affinity import *
from openai import RateLimitError, APIConnectionError, APIError
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from cap.lmp.utils import load_prompt


def _strip_echoed_query(text, use_query):
    """Strip the echoed query line from the start of model output.

    The Responses API models often echo back the query/prompt prefix,
    sometimes with minor rephrasing (e.g. "screw driver" → "screwdriver").
    We strip it by:
    1. Exact prefix match (original behaviour).
    2. If the query has a known prefix pattern (e.g. "# Query: "), strip any
       first line that starts with that prefix — handles rephrased echoes.
    """
    stripped = text.strip()
    # Exact match
    if use_query and stripped.startswith(use_query.strip()):
        stripped = stripped[len(use_query.strip()) :].strip()
        return stripped
    # Fuzzy: strip any leading line that starts with the query prefix
    # (e.g. "# Query: " or "# "). This handles rephrased echoes.
    if use_query:
        # Find the prefix pattern: everything up to and including ": " or just "# "
        prefix = use_query.strip()
        # Try common prefixes like "# Query: ", "# ", "# define function: "
        for sep in [": ", " "]:
            idx = prefix.find(sep)
            if idx != -1:
                prefix_pattern = prefix[: idx + len(sep)]
                if stripped.startswith(prefix_pattern):
                    # Strip everything up to the first newline
                    newline_idx = stripped.find("\n")
                    if newline_idx != -1:
                        stripped = stripped[newline_idx + 1 :].strip()
                    else:
                        stripped = ""  # entire response was just the echo
                    return stripped
    return stripped


def _trim_at_stop_tokens(text, stop_tokens):
    """Trim text at the first occurrence of any stop token.

    The Responses API does not support stop tokens natively,
    so we handle them in post-processing.
    """
    if not stop_tokens:
        return text
    earliest = len(text)
    for token in stop_tokens:
        idx = text.find(token)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest]


def _build_openai_client(cfg):
    """Build OpenAI-compatible client from LMP config.

    Supports both OpenAI-hosted models and OpenAI-compatible OSS servers.
    """
    base_url = cfg.get("base_url", None)
    api_key = cfg.get("api_key", None)

    if base_url:
        if api_key is None:
            api_key = "not-needed"
        return OpenAI(base_url=base_url, api_key=api_key)

    if api_key is not None:
        return OpenAI(api_key=api_key)

    return OpenAI()


def _request_model_text(client, cfg, prompt, instructions):
    """Request model output text using configured API mode."""
    model = cfg.get("model", "gpt-5-nano")
    api_mode = cfg.get("api_mode", "responses")

    if api_mode == "chat_completions":
        messages = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": cfg["max_tokens"],
        }
        if "temperature" in cfg:
            kwargs["temperature"] = cfg["temperature"]

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or "", response

    kwargs = {
        "model": model,
        "instructions": instructions,
        "input": prompt,
        "max_output_tokens": cfg["max_tokens"],
    }

    reasoning_effort = cfg.get("reasoning_effort", "low")
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**kwargs)
    return response.output_text or "", response


class LMP:
    def __init__(self, name, cfg, lmp_fgen, fixed_vars, variable_vars):
        self._name = name
        self._cfg = cfg
        self._client = _build_openai_client(cfg)

        env = "real"

        self._base_prompt = load_prompt(f"{env}/{self._cfg['prompt_fname']}.txt")

        self._stop_tokens = list(self._cfg["stop"])

        self._lmp_fgen = lmp_fgen

        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ""

    def clear_exec_hist(self):
        self.exec_hist = ""

    def build_prompt(self, query, context=""):
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = (
                f"from utils import {', '.join(self._variable_vars.keys())}"
            )
        else:
            variable_vars_imports_str = ""
        prompt = self._base_prompt.replace(
            "{variable_vars_imports}", variable_vars_imports_str
        )

        if self._cfg["maintain_session"]:
            prompt += f"\n{self.exec_hist}"

        if context != "":
            prompt += f"\n{context}"

        use_query = f"{self._cfg['query_prefix']}{query}{self._cfg['query_suffix']}"
        prompt += f"\n{use_query}"

        return prompt, use_query

    def _generate_code_from_prompt(self, prompt, use_query):
        while True:
            try:
                raw_text, response = _request_model_text(
                    self._client,
                    self._cfg,
                    prompt,
                    "You are a helpful assistant that pays attention to the user's instructions and writes good python code for operating a robot arm in a tabletop environment. Only output python code with no explanation (code comments are ok) or formatting, it should be ready to parse directly and ran. Do not repeat my code, just complete/continue from my code. Do not import any packages that aren't already there, you should never use the import keyword since all packages you need are already imported in the examples.",
                )

                status = getattr(response, "status", "n/a")
                print(
                    f"[DEBUG LMP {self._name}] status={status}, output_text length={len(raw_text)}"
                )
                print(f"[DEBUG LMP {self._name}] raw response:\n---\n{raw_text}\n---")

                # Strip markdown code fences if model wraps output
                stripped = raw_text.strip()
                if stripped.startswith("```python"):
                    stripped = stripped[len("```python") :].strip()
                if stripped.startswith("```"):
                    stripped = stripped[len("```") :].strip()
                if stripped.endswith("```"):
                    stripped = stripped[:-3].strip()

                # Strip echoed query prefix (Responses API models echo it back)
                stripped = _strip_echoed_query(stripped, use_query)

                code_str = _trim_at_stop_tokens(stripped, self._stop_tokens)
                print(f"[DEBUG LMP {self._name}] after trim:\n---\n{code_str}\n---")
                return code_str
            except (RateLimitError, APIConnectionError, APIError) as e:
                print(f"OpenAI API got err {e}")
                print("Retrying after 10s.")
                sleep(10)

    def generate_code(self, query, context=""):
        """Generate LMP code without executing it."""
        prompt, use_query = self.build_prompt(query, context=context)
        print(
            f"[DEBUG LMP {self._name}] prompt length: {len(prompt)} chars, use_query: {use_query!r}"
        )
        code_str = self._generate_code_from_prompt(prompt, use_query)
        return {
            "prompt": prompt,
            "use_query": use_query,
            "code": code_str,
        }

    def __call__(self, query, context="", **kwargs):
        prompt, use_query = self.build_prompt(query, context=context)
        print(
            f"[DEBUG LMP {self._name}] prompt length: {len(prompt)} chars, use_query: {use_query!r}"
        )
        code_str = self._generate_code_from_prompt(prompt, use_query)

        if self._cfg["include_context"] and context != "":
            to_exec = f"{context}\n{code_str}"
            to_log = f"{context}\n{use_query}\n{code_str}"
        else:
            to_exec = code_str
            to_log = f"{use_query}\n{to_exec}"

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())
        print(f"LMP {self._name} exec:\n\n{to_log_pretty}\n")

        new_fs = self._lmp_fgen.create_new_fs_from_code(code_str)
        self._variable_vars.update(new_fs)

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        if not self._cfg["debug_mode"]:
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f"\n{to_exec}"

        if self._cfg["maintain_session"]:
            self._variable_vars.update(lvars)

        if self._cfg["has_return"]:
            return lvars[self._cfg["return_val_name"]]


class LMPFGen:
    def __init__(self, cfg, fixed_vars, variable_vars):
        self._cfg = cfg
        self._client = _build_openai_client(cfg)

        self._stop_tokens = list(self._cfg["stop"])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars

        self._base_prompt = self._cfg["prompt_fname"]

    def create_f_from_sig(
        self, f_name, f_sig, other_vars=None, fix_bugs=False, return_src=False
    ):
        print(f"Creating function: {f_sig}")

        use_query = f"{self._cfg['query_prefix']}{f_sig}{self._cfg['query_suffix']}"
        prompt = f"{self._base_prompt}\n{use_query}"

        while True:
            try:
                raw_text, _response = _request_model_text(
                    self._client,
                    self._cfg,
                    prompt,
                    "You are a helpful coding assistant. Only output python code with no explanation (code comments are ok) or formatting, it should be ready to parse directly and ran.",
                )
                print(f"[DEBUG LMPFGen] raw response:\n---\n{raw_text}\n---")

                stripped = raw_text.strip()
                if stripped.startswith("```python"):
                    stripped = stripped[len("```python") :].strip()
                if stripped.startswith("```"):
                    stripped = stripped[len("```") :].strip()
                if stripped.endswith("```"):
                    stripped = stripped[:-3].strip()

                stripped = _strip_echoed_query(stripped, use_query)

                f_src = _trim_at_stop_tokens(stripped, self._stop_tokens)
                break
            except (RateLimitError, APIConnectionError, APIError) as e:
                print(f"OpenAI API got err {e}")
                print("Retrying after 10s.")
                sleep(10)

        if fix_bugs:
            try:
                edit_text, _edit_response = _request_model_text(
                    self._client,
                    self._cfg,
                    f_src,
                    "Fix any bugs in the following code. Improve readability. Keep same inputs and outputs. Only make small changes. No comments. Only output python code with no explanation (code comments are ok) or formatting, it should be ready to parse directly and ran.",
                )
                f_src = edit_text.strip()
            except Exception as e:
                print(f"Bug fixing failed: {e}. Using original code.")

        if other_vars is None:
            other_vars = {}
        gvars = merge_dicts([self._fixed_vars, self._variable_vars, other_vars])
        lvars = {}

        exec_safe(f_src, gvars, lvars)

        f = lvars[f_name]

        to_print = highlight(
            f"{use_query}\n{f_src}", PythonLexer(), TerminalFormatter()
        )
        print(f"LMP FGEN created:\n\n{to_print}\n")

        if return_src:
            return f, f_src
        return f

    def create_new_fs_from_code(
        self, code_str, other_vars=None, fix_bugs=False, return_src=False
    ):
        fs, f_assigns = {}, {}
        f_parser = FunctionParser(fs, f_assigns)
        f_parser.visit(ast.parse(code_str))
        for f_name, f_assign in f_assigns.items():
            if f_name in fs:
                fs[f_name] = f_assign

        if other_vars is None:
            other_vars = {}

        new_fs = {}
        srcs = {}
        for f_name, f_sig in fs.items():
            all_vars = merge_dicts(
                [self._fixed_vars, self._variable_vars, new_fs, other_vars]
            )
            if not var_exists(f_name, all_vars):
                f, f_src = self.create_f_from_sig(
                    f_name, f_sig, new_fs, fix_bugs=fix_bugs, return_src=True
                )

                # recursively define child_fs in the function body if needed
                f_def_body = astunparse.unparse(ast.parse(f_src).body[0].body)
                child_fs, child_f_srcs = self.create_new_fs_from_code(
                    f_def_body, other_vars=all_vars, fix_bugs=fix_bugs, return_src=True
                )

                if len(child_fs) > 0:
                    new_fs.update(child_fs)
                    srcs.update(child_f_srcs)

                    # redefine parent f so newly created child_fs are in scope
                    gvars = merge_dicts(
                        [self._fixed_vars, self._variable_vars, new_fs, other_vars]
                    )
                    lvars = {}

                    exec_safe(f_src, gvars, lvars)

                    f = lvars[f_name]

                new_fs[f_name], srcs[f_name] = f, f_src

        if return_src:
            return new_fs, srcs
        return new_fs


class FunctionParser(ast.NodeTransformer):
    def __init__(self, fs, f_assigns):
        super().__init__()
        self._fs = fs
        self._f_assigns = f_assigns

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            f_sig = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.func).strip()
            self._fs[f_name] = f_sig
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call):
            assign_str = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.value.func).strip()
            self._f_assigns[f_name] = assign_str
        return node


def var_exists(name, all_vars):
    try:
        eval(name, all_vars)
    except:
        exists = False
    else:
        exists = True
    return exists


def merge_dicts(dicts):
    return {k: v for d in dicts for k, v in d.items()}


def exec_safe(code_str, gvars=None, lvars=None):
    # This is problematic, fix later. Removing this check is unsafe but makes the code crash less
    # banned_phrases = ['import', '__']
    # for phrase in banned_phrases:
    #     assert phrase not in code_str

    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}

    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([gvars, {"exec": empty_fn, "eval": empty_fn}])
    exec(code_str, custom_gvars, lvars)
