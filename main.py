from cap.lmp.utils import load_config
from cap.lmp.lmp_wrapper import setup_LMP

# Setup
config = load_config('cap/config/real_config.yaml')
lmp_tabletop_ui, lmp_env = setup_LMP(config)

# Add objects (replace with actual computer vision)
lmp_env.update_object_list(['red block', 'blue bowl'])

# Execute commands
lmp_tabletop_ui("put the red block on the blue bowl", f'objects = {lmp_env.get_obj_names()}')
