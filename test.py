from cap.lmp.lmp import LMP, LMPFGen
from cap.lmp.utils import load_config

config = load_config('cap/config/real_config.yaml')

lmp_ui_config = config["lmp_config"]['lmps']["tabletop_ui"]
lmp_fgen_config = config["lmp_config"]['lmps']["fgen"]

lmp_fgen = LMPFGen(lmp_fgen_config, {}, {})
lmp_tabletop_ui = LMP(
    'tabletop_ui', lmp_ui_config, lmp_fgen, {}, {}
)

lmp_tabletop_ui("Find the distance between (0, 0, 0) and (1, 1, 1)")
