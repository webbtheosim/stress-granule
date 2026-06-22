"""
Heterotypic Wang-Frenkel mixing-rule selection for the MPiPi force field.

Mixing helper for the small-molecule parameterisation pipeline, imported by
``gen_files.py`` (and exercised standalone via ``__main__``). It loads the
homotypic (like-like) Wang-Frenkel parameters for the protein/RNA reference
beads and the new small-molecule beads, evaluates several combining rules for
each heterotypic (unlike) parameter, and either picks the rule that best
reproduces the reference cross-interactions (``WF_Parameters.txt``) or applies a
caller-supplied override. The result is a full per-pair interaction table.

Combining rules evaluated per parameter:
    epsilon: Lorentz-Berthelot (geometric), Waldman-Hagler, Fender-Halsey, Kong
    sigma:   Lorentz-Berthelot (arithmetic), Waldman-Hagler, Kong
    v/mu/rc: geometric vs arithmetic mean

Inputs: small-molecule parameter CSV (homotypic block from row 25 on) and the
reference WF cross-parameter file.
Outputs (instance attributes): ``sm_parameters`` (nested ``{i: {j: {eps, sig,
v, mu, rc}}}`` interaction table for j >= i) and ``used_rules`` (the rule chosen
per parameter), populated by :meth:`calc_sm_mixing`.

Not a standalone tool in the documented workflow; instantiate ``mixer`` from the
pipeline.
"""

import os

import numpy as np
from sklearn.metrics import mean_squared_error

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class mixer:
    """Select and apply Wang-Frenkel combining rules for heterotypic pairs."""

    def __init__(self, csv_file, parameters_file=os.path.join(_DATA_DIR, "WF_Parameters.txt"), rules_override=None):
        """Load homotypic reference and small-molecule parameters.

        Args:
            csv_file: Small-molecule parameter CSV; the homotypic block is read
                from row 25 onward (``name, num, eps, sig, v, mu, rc, ...``).
            parameters_file: Reference WF cross-parameter file used to score the
                combining rules (``WF_Parameters.txt``).
            rules_override: Optional ``{param: rule}`` dict forcing a specific
                combining rule per parameter instead of auto-selecting the best.
        """
        self.parameters_file = parameters_file
        self.csv_file = csv_file
        self.homotypic_actual_parameters = {}
        self.heterotypic_actual_parameters = {}
        self.pred_parameters = {}
        self.actual_parameters = {}
        self.sm_homotypic_parameters = {}
        self.sm_parameters = {}
        self.homotypic = {}
        self.parse_file()
        self.parse_csv()
        self.all_params = {}
        self.rules_override = rules_override
        self.used_rules = None

    def parse_file(self):
        """Load reference WF parameters into homo-/heterotypic lookup tables."""
        with open(self.parameters_file, 'r') as params:
            lines = params.readlines()
            for line in lines:
                line_split = line.split()
                atom1 = int(line_split[1])
                atom2 = int(line_split[2])
                eps = float(line_split[4])
                sig = float(line_split[5])
                v = int(line_split[6])
                mu = int(line_split[7])
                rc = float(line_split[8])

                if atom1 in self.actual_parameters.keys():
                    self.actual_parameters[atom1][atom2] = {"eps": eps, "sig": sig, "v": v, "mu": mu, "rc": rc}
                else:
                    self.actual_parameters[atom1] = {atom2: {"eps": eps, "sig": sig, "v": v, "mu": mu, "rc": rc}}

                if atom1 == atom2:
                    self.homotypic_actual_parameters[atom1] = {
                        atom2: {"eps": eps, "sig": sig, "v": v, "mu": mu, "rc": rc}}
                else:
                    if atom1 in self.heterotypic_actual_parameters.keys():
                        self.heterotypic_actual_parameters[atom1][atom2] = {"eps": eps, "sig": sig, "v": v, "mu": mu,
                                                                            "rc": rc}
                    else:
                        self.heterotypic_actual_parameters[atom1] = {
                            atom2: {"eps": eps, "sig": sig, "v": v, "mu": mu, "rc": rc}}

    def parse_csv(self):
        """Load small-molecule homotypic parameters and build the homotypic table.

        Merges the reference and small-molecule homotypic parameters into
        ``self.homotypic``, the per-bead source used when applying mixing rules.
        """
        with open(self.csv_file, 'r') as params:
            lines = params.readlines()[25:]
            for line in lines:
                data = line.split(",")
                atom = int(data[1])
                eps = float(data[2])
                sig = float(data[3])
                v = int(data[4])
                mu = int(data[5])
                rc = float(data[6])
                self.sm_homotypic_parameters[atom] = {atom: {"eps": eps, "sig": sig, "v": v, "mu": mu, "rc": rc}}

        for key in self.homotypic_actual_parameters:
            self.homotypic[key] = (self.homotypic_actual_parameters[key][key])

        for key in self.sm_homotypic_parameters:
            self.homotypic[key] = (self.sm_homotypic_parameters[key][key])


    def geometric_mix(self, par1, par2):
        """Geometric mean ``sqrt(par1 * par2)`` (Lorentz-Berthelot epsilon)."""
        mix = np.sqrt(par1 * par2)
        return mix

    def arithmetic_mix(self, par1, par2):
        """Arithmetic mean ``(par1 + par2) / 2`` (Lorentz-Berthelot sigma)."""
        mix = (par1 + par2) / 2
        return mix

    def wh_mix_s(self, s1, s2):
        """Waldman-Hagler combined sigma from two homotypic sigmas."""
        mix = np.power((np.power(s1, 6) + np.power(s2, 6)) / 2, 1 / 6)
        return mix

    def wh_mix_e(self, e1, e2, s1, s2):
        """Waldman-Hagler combined epsilon from two epsilon/sigma pairs."""
        mix = 2 * np.sqrt(e1 * e2) * ((np.power(s1, 3) * np.power(s2, 3)) / (np.power(s1, 6) + np.power(s2, 6)))
        return mix

    def fh_mix_e(self, e1, e2):
        """Fender-Halsey combined epsilon (harmonic mean of epsilons)."""
        mix = (2 * e1 * e2) / (e1 + e2)
        return mix

    def fh_mix_s(self, s1, s2):
        """Fender-Halsey combined sigma (arithmetic mean of sigmas)."""
        mix = self.arithmetic_mix(s1, s2)
        return mix

    def k_mix_s(self, e1, e2, s1, s2):
        """Kong combined sigma from two epsilon/sigma pairs."""
        mix = np.power(
            np.power((np.power((e1 * np.power(s1, 12)), 1 / 13)
                      + np.power((e2 * np.power(s2, 12)), 1 / 13)) / 2, 13)
            , 1 / 6)
        return mix

    def k_mix_e(self, e1, e2, s1, s2):
        """Kong combined epsilon (uses :meth:`k_mix_s` for the sigma term)."""
        mix = (e1*np.power(s1, 6)*e2*np.power(s2, 6))/np.power(self.k_mix_s(e1, e2, s1, s2), 6)
        return mix

    def calc_rmse(self, par_test, par_pred):
        """Return the RMSE between reference and predicted parameter vectors."""
        rmse = np.sqrt(mean_squared_error(par_test, par_pred))
        return rmse

    def get_test_params(self, param):
        """Collect the reference heterotypic values of ``param`` across all pairs."""
        params = []
        for key1 in self.actual_parameters.keys():
            for key2 in self.actual_parameters[key1].keys():
                params.append(self.actual_parameters[key1][key2][param])
        return params

    def calc_pred_params(self, param):
        """Predict heterotypic ``param`` for every pair under each combining rule.

        Returns a dict keyed by rule label (e.g. ``LB``/``WH``/``FH``/``K`` for
        eps/sig, ``G``/``A`` for v/mu/rc), each holding the per-pair predictions
        in the same pair order as :meth:`get_test_params`.
        """
        mix_params = {"eps": {"LB": [], "WH": [], "FH": [], "K": []}, "sig": {"LB": [], "WH": [], "FH": [], "K": []}, "v": {"G": [], "A": []}, "mu": {"G": [], "A": []}, "rc": {"G": [], "A": []}}

        for key1 in self.actual_parameters.keys():
            for key2 in self.actual_parameters[key1].keys():
                param1 = self.homotypic_actual_parameters[key1][key1][param]
                param2 = self.homotypic_actual_parameters[key2][key2][param]

                if param == "eps":
                    param3 = self.homotypic_actual_parameters[key1][key1]["sig"]
                    param4 = self.homotypic_actual_parameters[key2][key2]["sig"]

                    mix_params["eps"]["LB"].append(self.geometric_mix(param1, param2))
                    mix_params["eps"]["WH"].append(self.wh_mix_e(param1, param2, param3, param4))
                    mix_params["eps"]["FH"].append(self.fh_mix_e(param1, param2))
                    mix_params["eps"]["K"].append(self.k_mix_e(param1, param2, param3, param4))

                elif param == "sig":
                    param3 = self.homotypic_actual_parameters[key1][key1]["eps"]
                    param4 = self.homotypic_actual_parameters[key2][key2]["eps"]

                    mix_params["sig"]["LB"].append(self.arithmetic_mix(param1, param2))
                    mix_params["sig"]["WH"].append(self.wh_mix_s(param1, param2))
                    mix_params["sig"]["FH"].append(self.arithmetic_mix(param1, param2))
                    mix_params["sig"]["K"].append(self.k_mix_e(param3, param4, param1, param2))

                elif param == "v":
                    mix_params["v"]["G"].append(self.geometric_mix(param1, param2))
                    mix_params["v"]["A"].append(self.arithmetic_mix(param1, param2))

                elif param == "mu":
                    mix_params["mu"]["G"].append(self.geometric_mix(param1, param2))
                    mix_params["mu"]["A"].append(self.arithmetic_mix(param1, param2))

                elif param == "rc":
                    mix_params["rc"]["G"].append(self.geometric_mix(param1, param2))
                    mix_params["rc"]["A"].append(self.arithmetic_mix(param1, param2))

        return mix_params

    def compare_mixing_models(self):
        """Pick the lowest-RMSE combining rule for each Wang-Frenkel parameter.

        Scores every rule's predictions against the reference heterotypic values
        and returns ``{param: best_rule_label}``.
        """
        mixing_parameters = ["eps", "sig", "v", "mu", "rc"]
        optimal_model = {}

        for parameter in mixing_parameters:
            test = self.get_test_params(parameter)
            prediction = self.calc_pred_params(parameter)
            scores = {}
            for mod in prediction[parameter].keys():
                pred = prediction[parameter][mod]
                score = self.calc_rmse(test, pred)
                scores[mod] = score
            optimal_model[parameter] = min(scores, key=scores.get)
            print(scores)

        print(optimal_model)
        return optimal_model


    def calc_sm_mixing(self):
        """Build the full heterotypic interaction table in ``self.sm_parameters``.

        Selects the per-parameter combining rule (override or best-fit), then
        fills ``self.sm_parameters[i][j]`` (for j >= i) with the reference values
        for existing MPiPi-MPiPi pairs and with mixed values for any pair
        involving a small-molecule bead. Records the chosen rules in
        ``self.used_rules``.
        """
        if self.rules_override:
            optimal_model = self.rules_override
        else:
            optimal_model = self.compare_mixing_models()
        self.used_rules = optimal_model.copy()
        mixing_parameters = ["eps", "sig", "v", "mu", "rc"]
        parameter_dict = {}
        for key1 in self.homotypic.keys():
            for key2 in self.homotypic.keys():
                for param in mixing_parameters:

                    value1 = self.homotypic[key1][param]
                    value2 = self.homotypic[key2][param]

                    if param == "eps":
                        value3 = self.homotypic[key1]["sig"]
                        value4 = self.homotypic[key2]["sig"]
                        if optimal_model[param] == "LB":
                            parameter_dict[param] = self.geometric_mix(value1, value2)
                        elif optimal_model[param] == "WH":
                            parameter_dict[param] = self.wh_mix_e(value1, value2, value3, value4)
                        elif optimal_model[param] == "FH":
                            parameter_dict[param] = self.fh_mix_e(value1, value2)
                        elif optimal_model[param] == "K":
                            parameter_dict[param] = self.k_mix_e(value1, value2, value3, value4)

                    elif param == "sig":
                        value3 = self.homotypic[key1]["eps"]
                        value4 = self.homotypic[key2]["eps"]
                        if optimal_model[param] == "LB":
                            parameter_dict[param] = self.arithmetic_mix(value1, value2)
                        elif optimal_model[param] == "WH":
                            parameter_dict[param] = self.wh_mix_s(value1, value2)
                        elif optimal_model[param] == "FH":
                            parameter_dict[param] = self.arithmetic_mix(value1, value2)
                        elif optimal_model[param] == "K":
                            parameter_dict[param] = self.k_mix_s(value3, value4, value1, value2)

                    elif param == "rc":
                        if optimal_model[param] == "G":
                            parameter_dict[param] = (self.geometric_mix(value1, value2))
                        elif optimal_model[param] == "A":
                            parameter_dict[param] = (self.arithmetic_mix(value1, value2))

                    else:
                        if optimal_model[param] == "G":
                            parameter_dict[param] = int(np.round(self.geometric_mix(value1, value2)))
                        elif optimal_model[param] == "A":
                            parameter_dict[param] = int(np.round(self.arithmetic_mix(value1, value2)))

                if int(key2) >= int(key1):
                    if key1 in range(1, 25) and key2 in range(1, 25):
                        if key1 in self.sm_parameters.keys():
                            self.sm_parameters[key1][key2] = self.actual_parameters[key1][key2]
                        else:
                            self.sm_parameters[key1] = {key2: self.actual_parameters[key1][key2]}


                    else:
                        if key1 in self.sm_parameters.keys():
                            self.sm_parameters[key1][key2] = {"eps": parameter_dict["eps"], "sig": parameter_dict["sig"], "v": parameter_dict["v"], "mu": parameter_dict["mu"], "rc": parameter_dict["rc"]}
                        else:
                            self.sm_parameters[key1] = {key2: {"eps": parameter_dict["eps"], "sig": parameter_dict["sig"], "v": parameter_dict["v"], "mu": parameter_dict["mu"], "rc": parameter_dict["rc"]}}




if __name__ == '__main__':
    # Smoke test: build the mixer from the default parameter table, select the
    # combining rules (prints per-parameter RMSE scores + chosen rules), and
    # report the size of the resulting heterotypic interaction table.
    parameter_file = "parameters.csv"
    mixer_obj = mixer(parameter_file)
    mixer_obj.calc_sm_mixing()

    count = sum(len(pairs) for pairs in mixer_obj.sm_parameters.values())
    print(f"Generated {count} heterotypic parameter pairs")
