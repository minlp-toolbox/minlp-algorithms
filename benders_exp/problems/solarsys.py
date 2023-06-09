import casadi as ca
from benders_exp.problems.dsc import Description

def create_solarsys_model():
    #dsc = Description()
    #T_hts = dsc.sym("T_hts", 4, 0, ca.inf)
    #T_lts = dsc.sym("T_lts", 1, 0, ca.inf)
    #T_fpsc = dsc.sym("T_fpsc", 1, 0, ca.inf)
    #T_fpsc_s = dsc.sym("T_fpsc_s", 1, 0, ca.inf)
    #T_vtsc = dsc.sym("T_vtsc", 1, 0, ca.inf)
    #T_vtsc_s = dsc.sym("T_vtsc_s", 1, 0, ca.inf)
    #T_pscf = dsc.sym("T_pscf", 1, 0, ca.inf)
    #T_pscr = dsc.sym("T_pscr", 1, 0, ca.inf)
    #T_shx_psc = dsc.sym("T_shx_psc", 4, 0, ca.inf)
    #T_shx_ssc = dsc.sym("T_shx_ssc", 4, 0, ca.inf)

    #v_ppsc = dsc.sym("v_ppsc", 1, 0, ca.inf)
    #p_mpsc = dsc.sym("p_mpsc", 1, 0, ca.inf)
    #v_pssc = dsc.sym("v_pssc", 1, 0, ca.inf)
    #P_g = dsc.sym("P_g", 1, 0, ca.inf)
    #mdot_o_hts_b = dsc.sym("mdot_o_hts_b", 1, 0, ca.inf)
    #mdot_i_hts_b = dsc.sym("mdot_i_hts_b", 1, 0, ca.inf)

    #b_ac = dsc.sym_bool("b_ac", 1)
    #b_fc = dsc.sym_bool("b_fc", 1)
    #b_hp = dsc.sym_bool("b_hp", 1)

    #T_amb = dsc.add_parameters("T_amb", 1, 0)
    #I_fpsc = dsc.add_parameters("I_fpsc", 1, 0)
    #I_vtsc = dsc.add_parameters("I_vtsc", 1, 0)
    #Qdot_c = dsc.add_parameters("Qdot_c", 1, 0)
    #P_pv_kWp = dsc.add_parameters("P_pv_kWp", 1, 0)
    #p_g = dsc.add_parameters("p_g", 1, 0)

        self.x = ca.MX.sym("x", self.nx)
        self.b = ca.MX.sym("b", self.nb)
        self.u = ca.MX.sym("u", self.nu)
        self.c = ca.MX.sym("c", self.nc)
