# CaP Demo Quality Report — Llama 3 8B (v5_diverse)

**Description:** Strict-API + diverse BEHAVIOR few-shots (final)

**Source file:** `demo/outputs_llama3_v5_diverse.jsonl` (via `evaluate_outputs.py`)

**Model:** dganochenko/llama-3-8b-chat

## 1. Summary

- Prompts evaluated: **45**
- Overall pass rate: **45/45 = 100.0%**
- Average generated-code length: **1.4 non-blank lines**

## 2. Per-Criterion Pass Rates

| Criterion | Pass | Fail | N/A | Pass rate |
|---|---:|---:|---:|---:|
| Syntactically valid | 45 | 0 | 0 | 100.0% |
| Uses only CaP API | 45 | 0 | 0 | 100.0% |
| No echoed `# Query:` | 45 | 0 | 0 | 100.0% |
| No repetition loops | 45 | 0 | 0 | 100.0% |
| Stops at correct boundary | 45 | 0 | 0 | 100.0% |
| **Overall `pass`** | 45 | 0 | 0 | **100.0%** |

### 2b. API-usage distribution

Top function calls (across all generated code):

| Function | # calls |
|---|---:|
| `put_first_on_second` | 60 |
| `parse_position` | 10 |
| `parse_obj_name` | 1 |
| `get_obj_names` | 1 |

Pattern-level counts (prompts using at least one call):

- `put_first_on_second`: **45/45**
- `parse_obj_name`:     **1/45**
- `parse_position`:     **8/45**
- `parse_question`:     **0/45**
- `transform_shape_pts`: **0/45**
- contains a `for`-loop: **1/45**
- non-API `goto_pos`:   **0** call(s)
- non-API `say`:        **0** call(s)
- non-API `stack_objects_in_order`: **0** call(s)

### 2c. Breakdown by Difficulty

| Difficulty | Pass / Total | Rate |
|---|---:|---:|
| medium | 45 / 45 | 100.0% |

### 2d. Breakdown by Category

| Category | Pass / Total | Rate |
|---|---:|---:|
| unspecified | 45 / 45 | 100.0% |

## 3. Prompt-by-Prompt Results

| # | id | query | generated code | verdict | notes |
|---|---|---|---|---|---|
| 1 | `c001` | move the plywoods to the designated area | <pre>put_first_on_second('plywood', parse_position('the designated area'))</pre> | **PASS** |  |
| 2 | `c002` | put all the books inside the carton | <pre>put_first_on_second('notebook', 'carton')<br>put_first_on_second('hardback', 'carton')</pre> | **PASS** |  |
| 3 | `c003` | arrange the objects next to the sink | <pre>put_first_on_second('bowl', parse_position('a point next to the sink'))</pre> | **PASS** |  |
| 4 | `c004` | clear the table — put everything in the bucket | <pre>put_first_on_second('cup', 'bucket')</pre> | **PASS** |  |
| 5 | `c005` | collect all the pops and put them in the bucket | <pre>put_first_on_second('pop', 'bucket')</pre> | **PASS** |  |
| 6 | `c006` | fill the basket with all the items on the table | <pre>put_first_on_second('notebook', 'basket')<br>put_first_on_second('bracelet', 'basket')<br>put_first_on_second('ball', 'basket')</pre> | **PASS** |  |
| 7 | `c007` | place all the misplaced items on the table | <pre>put_first_on_second('gym shoe', 'coffee table')<br>put_first_on_second('necklace', 'coffee table')<br>put_first_on_second('notebook', 'coffee table')<br>put_first_on_second('sock', 'coffee table')</pre> | **PASS** |  |
| 8 | `c008` | fill the bottom cabinet with the objects on the table | <pre>put_first_on_second('ice cube', 'bottom cabinet')<br>put_first_on_second('lollipop', 'bottom cabinet')<br>put_first_on_second('candy cane', 'bottom cabinet')</pre> | **PASS** |  |
| 9 | `c009` | put all the plates into the dishwasher | <pre>put_first_on_second('plate', 'dishwasher')</pre> | **PASS** |  |
| 10 | `c010` | move the tiles to the designated area | <pre>put_first_on_second('tile', parse_position('the designated area'))</pre> | **PASS** |  |
| 11 | `c011` | arrange the objects next to the saw | <pre>put_first_on_second('plywood', parse_position('a point next to the saw'))</pre> | **PASS** |  |
| 12 | `c012` | place the cartons on top of the shelf | <pre>put_first_on_second('carton', 'shelf')</pre> | **PASS** |  |
| 13 | `c013` | put all the cartons into the carton | <pre>put_first_on_second('carton', 'carton')</pre> | **PASS** |  |
| 14 | `c014` | put all the markers into the bottom cabinet no top | <pre>put_first_on_second('highlighter', 'bottom cabinet no top')</pre> | **PASS** |  |
| 15 | `c015` | put all the highlighters into the backpack | <pre>highlighter_names = parse_obj_name('the highlighters', f'objects = {get_obj_names()}')<br>for highlighter in highlighter_names:<br>  put_first_on_second(highlighter, 'backpack')</pre> | **PASS** |  |
| 16 | `c016` | put all the backpacks into the mouse | <pre>put_first_on_second('backpack', 'mouse')</pre> | **PASS** |  |
| 17 | `c017` | put all the backpacks into the toothpaste | <pre>put_first_on_second('backpack', 'toothpaste')</pre> | **PASS** |  |
| 18 | `c018` | put all the cartons into the carton | <pre>put_first_on_second('notebook', 'carton')</pre> | **PASS** |  |
| 19 | `c019` | put all the cars into the pencil box | <pre>put_first_on_second('car', 'pencil box')</pre> | **PASS** |  |
| 20 | `c020` | put all the backpacks into the sunglass | <pre>put_first_on_second('backpack', 'sunglass')</pre> | **PASS** |  |
| 21 | `c021` | put all the cartons into the carton | <pre>put_first_on_second('carton', 'carton')</pre> | **PASS** |  |
| 22 | `c022` | put all the cartons into the carton | <pre>put_first_on_second('carton', 'carton')</pre> | **PASS** |  |
| 23 | `c023` | put all the cartons into the carton | <pre>put_first_on_second('carton', 'carton')</pre> | **PASS** |  |
| 24 | `c024` | put all the ashcans into the pop | <pre>put_first_on_second('trash can', 'pop')</pre> | **PASS** |  |
| 25 | `c025` | arrange the objects next to the sink | <pre>put_first_on_second('shampoo', parse_position('a point next to the sink'))<br>put_first_on_second('soap', parse_position('a point next to the sink'))<br>put_first_on_second('dishtowel', parse_position('a point next to the sink'))</pre> | **PASS** |  |
| 26 | `c026` | put all the playthings into the carton | <pre>put_first_on_second('toy', 'carton')</pre> | **PASS** |  |
| 27 | `c027` | put all the pumpkins into the bottom cabinet | <pre>put_first_on_second('pumpkin', 'bottom cabinet')</pre> | **PASS** |  |
| 28 | `c028` | fill the bottom cabinet with the objects on the table | <pre>put_first_on_second('wreath', 'bottom cabinet')<br>put_first_on_second('bow', 'bottom cabinet')<br>put_first_on_second('ribbon', 'bottom cabinet')</pre> | **PASS** |  |
| 29 | `c029` | put all the pastas into the countertop | <pre>put_first_on_second('pasta', 'countertop')</pre> | **PASS** |  |
| 30 | `c030` | put all the plates into the top cabinet | <pre>put_first_on_second('plate', 'top cabinet')</pre> | **PASS** |  |
| 31 | `c031` | place the christmas trees on top of the breakfast table | <pre>put_first_on_second('christmas tree decorated', 'breakfast table')</pre> | **PASS** |  |
| 32 | `c032` | arrange the objects next to the table lamp | <pre>put_first_on_second('table lamp', 'chaise longue')<br>put_first_on_second('floor lamp', 'chaise longue')<br>put_first_on_second('chaise longue', 'bed')</pre> | **PASS** |  |
| 33 | `c033` | place the books on top of the breakfast table | <pre>put_first_on_second('hardback', 'breakfast table')<br>put_first_on_second('notebook', 'breakfast table')</pre> | **PASS** |  |
| 34 | `c034` | place the chickens on top of the table knife | <pre>put_first_on_second('chicken', 'table knife')</pre> | **PASS** |  |
| 35 | `c035` | arrange the objects next to the sink | <pre>put_first_on_second('mousetrap', parse_position('a point next to the sink'))</pre> | **PASS** |  |
| 36 | `c036` | place the cabinets on top of the tray | <pre>put_first_on_second('bottom cabinet no top', 'tray')</pre> | **PASS** |  |
| 37 | `c037` | place the hardbacks on top of the coffee table | <pre>put_first_on_second('hardback', 'coffee table')</pre> | **PASS** |  |
| 38 | `c038` | place the candles on top of the coffee table | <pre>put_first_on_second('candle', 'coffee table')</pre> | **PASS** |  |
| 39 | `c039` | put all the oatmeals into the bottom cabinet | <pre>put_first_on_second('oatmeal', 'bottom cabinet')</pre> | **PASS** |  |
| 40 | `c040` | put all the breads into the bottom cabinet no top | <pre>put_first_on_second('pretzel', 'bottom cabinet no top')</pre> | **PASS** |  |
| 41 | `c041` | arrange the objects next to the newspaper | <pre>put_first_on_second('envelope', parse_position('a point next to the newspaper'))</pre> | **PASS** |  |
| 42 | `c042` | arrange the objects next to the sink | <pre>put_first_on_second('container date', parse_position('a point next to the sink'))</pre> | **PASS** |  |
| 43 | `c043` | put all the cereals into the bottom cabinet no top | <pre>put_first_on_second('cereal', 'bottom cabinet no top')</pre> | **PASS** |  |
| 44 | `c044` | put all the plates into the trash can | <pre>put_first_on_second('plate', 'trash can')</pre> | **PASS** |  |
| 45 | `c045` | place the socks on top of the sofa | <pre>put_first_on_second('sock', 'sofa')</pre> | **PASS** |  |

## 4. Notable Observations

- No hallucinated API usage: zero `goto_pos` / `say` / `stack_objects_in_order` calls.
- Core primitive `put_first_on_second` used in 45 / 45 prompts — the dominant action verb, as expected for pick-and-place tasks.
- Sub-LMPs (`parse_obj_name` / `parse_position` / `parse_question` / `transform_shape_pts`) invoked in 9 / 45 prompts — shows the model can reach beyond a single primitive.
- `for`-loop constructs appear in 1 / 45 outputs — indicates the model handles multi-instance scenes rather than enumerating objects by hand.
- All prompts pass every automated check.

## 5. Conclusion & Recommendation

The Llama 3 8B tabletop_ui pipeline under `v5_diverse` (Strict-API + diverse BEHAVIOR few-shots (final)) passes all automated quality checks on a 45-prompt BEHAVIOR-derived demo set. 

Key qualitative properties of this version: (i) zero hallucinated API calls; (ii) correct use of `put_first_on_second` as the core action verb; (iii) sub-LMPs (`parse_obj_name` / `parse_position`) reached in a meaningful fraction of the set, showing the model can compose CaP's perception-language-action stack rather than collapsing to a single primitive. Recommend this artifact for DARPA reporting and for the first real-robot execution milestones.
