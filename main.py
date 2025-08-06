from cap.lmp.lmp import LMP
from cap.lmp.utils import load_config

config = load_config('cap/config/real_config.yaml')

lmp_config = config["lmp_config"]['lmps']["writer"]

writer_lmp = LMP("writer_lmp", lmp_config, 
                 fixed_vars={}, 
                 variable_vars={}, 
                 debug=True, env='real')

print(writer_lmp("Write a function to calculate the factorial of a number."))
