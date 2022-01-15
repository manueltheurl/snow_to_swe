import pandas as pd
import numpy as np
from numpy import exp
import datetime as dt
np.seterr(divide='ignore', invalid='ignore')  # i do not really like that, but division by zero is going to nan silently


EARTH_GRAVITATIONAL_ACCELERATION = 9.81  # m/s^2


class SnowToSwe:
    def __init__(self, rho_max=401.2588, rho_null=81.19417, c_ov=0.0005104722, k_ov=0.37856737, k=0.02993175,
                 tau=0.02362476, eta_null=8523356):
        """
        This is a model to calculate the snow water equivalent out of given snow heights. The model is ported from R
        code which was written from Winkler et al. 2020 "Snow Water Equivalents exclusively from Snow Heights and their
        temporal Changes: The ∆SNOW.MODEL".

        The corresponding R model can be found here: https://r-forge.r-project.org/projects/nixmass/

        The code structure was adapted a bit but the calculations, available input and output variables stayed the same.

        Ported by Manuel Theurl who is taking no warranties for the correctness of the R to python port.

        Please refer to the official paper (lays in root of repo) for further description of the parameters.

        :param rho_max: Maximum Density
        :param rho_null: Fresh Snow Density
        :param c_ov: Overburden Parameter
        :param k_ov: Overburden Parameter
        :param k: Viscosity Parameter
        :param tau: Discrepancy Parameter
        :param eta_null: Viscosity Parameter
        """
        self.rho_max = rho_max
        self.rho_null = rho_null
        self.c_ov = c_ov
        self.k_ov = k_ov
        self.k = k
        self.tau = tau
        self.eta_null = eta_null

        self.prec = 10 ** -10  # precision for arithmetic comparisons [-]
        self._snowpack_dd = 0  # will get reset on every new convert
        self._current_day_info_string = ''  # information gets added freshly for every day

        print("This model is ported from R code which was written by Winkler et al. 2020 'Snow Water Equivalents "
              "exclusively from Snow Heights and their temporal Changes: The ∆SNOW.MODEL'")

    def convert_list(self, Hobs, timestep, verbose=False):
        """
        Converts a continuous time series of snow heights into snow water equivalents.

        :param Hobs:  List of snow heights evenly spaced by timestep
        :param timestep: in hours
        :param verbose: bool for printing out information about the ongoing process
        :return: List of swes on success else None
        """

        if any(np.isnan(Hobs)):
            print("swe.deltasnow: snow depth data must not be NA")
            return
        if not all([x >= 0 for x in Hobs]):
            print("swe.deltasnow: snow depth data must not be negative")
            return
        if not all([np.isreal(x) for x in Hobs]):
            print("swe.deltasnow: snow depth data must be numeric")
            return
        if Hobs[0]:
            print("swe.deltasnow: snow depth observations must start with 0")
            return

        ts = timestep * 3600  # timestep between observations [s]
        self._snowpack_dd = 0  # reset

        H = []  # modeled total height of snow at any day [m]
        SWE = []  # modeled total SWE at any day [kg/m2]
        ly = 1  # layer number [-]

        # preallocate matrix as days X layers
        ly_tot = np.count_nonzero(Hobs)  # maximum number of layers [-]
        day_tot = len(Hobs)  # total days from first to last snowfall [-]

        h = np.zeros((ly_tot, day_tot))  # modeled height of snow in all layers [m]
        swe = np.zeros((ly_tot, day_tot))  # modeled swe in all layers [kg/m2]
        age = np.zeros((ly_tot, day_tot))  # age of modeled layers [days]

        if verbose:
            print("Using parameters:")
            print("rho.max  =", self.rho_max)
            print("rho.null =", self.rho_null)
            print("c.ov     =", self.c_ov)
            print("k.ov     =", self.k_ov)
            print("k        =", self.k)
            print("tau      =", self.tau)
            print("eta.null =", self.eta_null)

        for t in range(day_tot):
            self._current_day_info_string = f"day {t+1}: "

            # snowdepth = 0, no snow cover
            if Hobs[t] == 0:
                if t > 0:
                    if Hobs[t - 1] != 0:
                        self._current_day_info_string += "runoff"

                try:  # actually brutally bad written, but whatever
                    H[t] = 0  # DIFFERENCE: H is a number, cannot index to it, in R you can
                    SWE[t] = 0
                except IndexError:
                    H.append(0)
                    SWE.append(0)
                h[:, t] = 0  # first column to 0
                swe[:, t] = 0
            # there is snow
            elif Hobs[t] > 0:  # redundant if, cause can snow height be negative?
                # first snow in/during season
                if Hobs[t - 1] == 0:
                    ly = 1
                    self._current_day_info_string += f"produce layer {ly} "
                    age[ly - 1, t] = 1
                    h[ly - 1, t] = Hobs[t]
                    H.append(Hobs[t])  # DIFFERENCE: H is a number, cannot index to it, in R you can
                    swe[ly - 1, t] = self.rho_null * Hobs[t]
                    SWE.append(swe[ly - 1, t])  # DIFFERENCE: SWE is a number, cannot index to it, in R you can

                    # compact actual day
                    snowpack_tomorrow = self.__dry_metamorphism(h[:, t], swe[:, t], age[:, t], ly_tot, ly, ts)

                    rl = self.__assignH(snowpack_tomorrow, h, swe, age, H, SWE, t, day_tot)
                    h = rl['h']
                    swe = rl["swe"]
                    age = rl["age"]
                    H = rl['H']
                    SWE = rl["SWE"]

                elif Hobs[t - 1] > 0:
                    deltaH = Hobs[t] - H[t]

                    if deltaH > self.tau:
                        self._current_day_info_string += f"create new layer {ly} "
                        sigma_null = deltaH * self.rho_null * EARTH_GRAVITATIONAL_ACCELERATION
                        epsilon = self.c_ov * sigma_null * exp(-self.k_ov * self._snowpack_dd["rho"] / (self.rho_max - self._snowpack_dd["rho"]))
                        h[:, t] = (1 - epsilon) * h[:, t]
                        # epsilon <- 1 - c.ov * sigma.null * exp(-k.ov * snowpack.dd$rho/(rho.max - snowpack.dd$rho))
                        # h[,t]     <- epsilon * h[,t]

                        swe[:, t] = swe[:, t - 1]
                        age[:ly, t] = age[:ly, t - 1] + 1

                        H[t] = sum(h[:, t])
                        SWE[t] = sum(swe[:, t])

                        # RHO[t]    <- SWE[t]/H[t]

                        # only for new layer
                        ly = ly + 1
                        h[ly - 1, t] = Hobs[t] - H[t]
                        swe[ly - 1, t] = self.rho_null * h[ly - 1, t]
                        age[ly - 1, t] = 1

                        # recompute
                        H[t] = sum(h[:, t])

                        SWE[t] = sum(swe[:, t])

                        # compact actual day
                        snowpack_tomorrow = self.__dry_metamorphism(h[:, t], swe[:, t], age[:, t], ly_tot, ly, ts)
                        # set values for next day
                        rl = self.__assignH(snowpack_tomorrow, h, swe, age, H, SWE, t, day_tot)
                        h = rl["h"]
                        swe = rl["swe"]
                        age = rl["age"]
                        H = rl["H"]
                        SWE = rl["SWE"]

                    # no mass gain or loss, but scaling
                    elif -self.tau <= deltaH <= self.tau:
                        self._current_day_info_string += "scaling: "
                        rl = self.__scaleH(t, ly, ly_tot, day_tot, deltaH, Hobs, h, swe, age, H, SWE, ts)
                        h = rl["h"]
                        swe = rl["swe"]
                        age = rl["age"]
                        H = rl["H"]
                        SWE = rl["SWE"]

                    elif deltaH < -self.tau:
                        self._current_day_info_string += "drenching: "
                        rl = self.__drenchH(t, ly, ly_tot, day_tot, Hobs, h, swe, age, H, SWE, ts)
                        h = rl["h"]
                        swe = rl["swe"]
                        age = rl["age"]
                        H = rl["H"]
                        SWE = rl["SWE"]

                    else:
                        self._current_day_info_string += "??"
            if verbose:
                print(self._current_day_info_string)
        return SWE

    def convert_csv(self, path_to_input_csv, path_to_output_csv=None, date_time_pattern="%Y-%m-%d", verbose=False):
        """
        Converts a continuous time series of snow heights given in a csv file into snow water equivalents.

        :param path_to_input_csv: path of a .csv file containing the data. Required columns are "date" with data format
            e.g. 2020-09-09 and "hs" (snow height in meters) with data format e.g. 0.56
        :param path_to_output_csv: Path where to save result csv, make sure all folders exist. If None it will not be
            saved in file.
        :param date_time_pattern: Pattern to convert date_time string to datetime object. Refer to python datetime
            strptime codes to find appropriate one
        :param verbose: bool for printing out information about the ongoing process
        :return: pandas time series with snow water equivalents
        """

        data = pd.read_csv(path_to_input_csv)
        try:
            dates = data["date"].tolist()
            time_resolution_in_seconds = self.__get_time_resolution_of_dates_in_seconds(dates, date_time_pattern)
            Hobs = data["hs"].tolist()

            if time_resolution_in_seconds:
                swes = self.convert_list(Hobs, time_resolution_in_seconds/3600, verbose=verbose)
                if swes is None:
                    print("SWE Conversion failed!")
                    return
                result_pandas_df = pd.DataFrame(list(zip(dates, swes)), columns=["date", "swe"])
                if path_to_output_csv is not None:
                    try:
                        result_pandas_df.to_csv(path_to_output_csv, float_format='%.3f')
                    except FileNotFoundError:
                        print(f"Cannot save csv result to {path_to_output_csv} .. make sure all folders exist!")
                return result_pandas_df
            else:
                print("Problems with time series dates")
                return None
        except KeyError:
            print("Wrong .csv file format! Required columns are 'date' with data format e.g. 2020-09-09 and 'hs' (snow "
                  "height in meters) with data format e.g. 0.56")
            return None

    @staticmethod
    def __assignH(sp_dd, h, swe, age, H, SWE, t, day_tot):
        if t < day_tot:
            h[:, t + 1] = sp_dd['h']
            swe[:, t + 1] = sp_dd["swe"]
            age[:, t + 1] = sp_dd["age"]
            H.append(sum(h[:, t + 1]))
            SWE.append(sum(swe[:, t + 1]))

        return {'h': h, "swe": swe, "age": age, 'H': H, "SWE": SWE}

    def __compactH(self, x, ts):
        # .d  -> today
        # .dd -> tomorrow
        age_d = 0 if x[0] == 0 else x[3]
        h_dd = x[0] / (1 + (x[2] * EARTH_GRAVITATIONAL_ACCELERATION * ts) / self.eta_null * exp(-self.k * x[1] / x[0]))
        h_dd = x[1] / self.rho_max if x[1] / h_dd > self.rho_max else h_dd
        h_dd = 0 if x[0] == 0 else h_dd
        swe_dd = x[1]
        age_dd = 0 if x[0] == 0 else age_d + 1
        rho_dd = 0 if x[0] == 0 else swe_dd / h_dd
        rho_dd = self.rho_max if self.rho_max - rho_dd < self.prec else rho_dd
        # return [h_dd, swe_dd, age_dd, rho_dd]
        # return x
        df = pd.DataFrame(columns=['h', "swe", "age", "rho"])
        df.loc[0] = [h_dd, swe_dd, age_dd, rho_dd]

        return pd.Series([h_dd, swe_dd, age_dd, rho_dd], index=['h', "swe", "age", "rho"])

    def __scaleH(self, t, ly, ly_tot, day_tot, deltaH, Hobs, h, swe, age, H, SWE, ts):
        # re-compact snowpack from yesterdays values with adapted eta
        # .d  -> yesterday
        # .dd -> today
        Hobs_d = Hobs[t - 1]

        Hobs_dd = Hobs[t]
        h_d = h[:, t - 1]
        swe_d = swe[:, t - 1]
        age_d = age[:, t]  # ; deltaH.d = deltaH

        # todays overburden
        swe_hat_d = []
        for i in range(ly_tot):
            swe_hat_d.append(sum(swe_d[i:ly_tot]))

        # analytical solution for layerwise adapted viskosity eta
        # assumption: recompaction ~ linear height change of yesterdays layers (see paper)
        eta_cor = []
        for i in range(ly_tot):
            rho_d = swe_d[i] / h_d[i]
            x = ts * EARTH_GRAVITATIONAL_ACCELERATION * swe_hat_d[i] * exp(-self.k * rho_d)  # yesterday
            P = h_d[i] / Hobs_d  # yesterday
            eta_i = Hobs_dd * x * P / (h_d[i] - Hobs_dd * P)
            eta_cor.append(0 if np.isnan(eta_i) else eta_i)

        # compute H of today with corrected eta
        # so that modeled H = Hobs
        h_dd_cor = np.array(h_d) / (1 + (np.array(swe_hat_d) * EARTH_GRAVITATIONAL_ACCELERATION * ts) / np.array(eta_cor) * exp(
            -self.k * np.array(swe_d) / np.array(h_d)))
        h_dd_cor[np.isnan(h_dd_cor)] = 0  # replace nan with 0
        H_dd_cor = sum(h_dd_cor)

        # and check, if Hd.cor is the same as Hobs.d
        if abs(H_dd_cor - Hobs_dd) > self.prec:
            self._current_day_info_string += f"WARNING: error in exponential re-compaction: H.dd.cor-Hobs.dd='{H_dd_cor - Hobs_dd}'"

        # which layers exceed rho.max?
        idx_max = []
        for i, (swe_e_val, h_dd_cor_val) in enumerate(zip(swe_d, h_dd_cor)):
            if swe_e_val / h_dd_cor_val - self.rho_max > self.prec:
                idx_max.append(i)

        # idx_max = np.where(, swe_d, h_dd_cor)[0]  # [0] cause tuple with list is returned
        if len(idx_max) > 0:
            if len(idx_max) < ly:
                # collect excess swe in those layers
                swe_excess = swe_d[idx_max] - h_dd_cor[idx_max] * self.rho_max

                # set affected layer(s) to rho.max
                swe_d[idx_max] = swe_d[idx_max] - swe_excess

                # distribute excess swe to other layers top-down
                lys = list(range(ly))
                for index in sorted(idx_max, reverse=True):
                    del lys[index]
                i = lys[len(lys) - 1]
                swe_excess_all = sum(swe_excess)

                while swe_excess_all > 0:
                    swe_res = h_dd_cor[i] * self.rho_max - swe_d[i]  # layer tolerates this swe amount to reach rho.max
                    if swe_res > swe_excess_all:
                        swe_res = swe_excess_all

                    swe_d[i] = swe_d[i] + swe_res
                    swe_excess_all = swe_excess_all - swe_res
                    i = i - 1
                    if i < 0 < swe_excess_all:
                        self._current_day_info_string += " runoff"
                        break
            else:
                # if all layers have density > rho.max
                # remove swe.excess from all layers (-> runoff)
                # (this sets density to rho.max)
                swe_excess = swe_d[idx_max] - h_dd_cor[idx_max] * self.rho_max
                swe_d[idx_max] = swe_d[idx_max] - swe_excess
                self._current_day_info_string += " runoff"

        h[:, t] = h_dd_cor
        swe[:, t] = swe_d
        age[:, t] = age_d
        H[t] = sum(h[:, t])
        SWE[t] = sum(swe[:, t])

        # compact actual day
        # if all layers already have maximum density rho_max
        # the snowpack will not be changed by the following step
        # nonlocal or not?????
        snowpack_tomorrow = self.__dry_metamorphism(h[:, t], swe[:, t], age[:, t], ly_tot, ly, ts)

        # set values for next day
        rl = self.__assignH(snowpack_tomorrow, h, swe, age, H, SWE, t, day_tot)
        h = rl["h"]
        swe = rl["swe"]
        age = rl["age"]
        H = rl["H"]
        SWE = rl["SWE"]

        return {'h': h, "swe": swe, "age": age, 'H': H, "SWE": SWE}

    @staticmethod
    def __get_time_resolution_of_dates_in_seconds(dates, date_time_pattern="%Y-%m-%d"):
        last_delta = None
        last_date = None
        for date_string in dates:
            try:
                current_date = dt.datetime.strptime(date_string, date_time_pattern)
            except ValueError:
                print(f"Wrong date_time pattern {date_time_pattern}! Refer to python datetime strptime codes to find "
                      f"appropriate!")
                return False
            if last_date is not None:
                current_delta = current_date - last_date

                if last_delta is not None and last_delta != current_delta:
                    print("Time series is not evenly spaced")
                    return False

                last_delta = current_delta
            last_date = current_date
        return last_delta.total_seconds()

    def __dry_metamorphism(self, h_d, swe_d, age_d, ly_tot, ly, ts):
        # h.d=h[,t];swe.d=swe[,t];age.d=age[,t]
        # snowpack.dd <- NULL
        # .d  -> today
        # .dd -> tomorrow

        # compute overburden for each layer
        # the overburden for the first layer is the layer itself

        swe_hat_d = []
        for i in range(ly_tot):
            swe_hat_d.append(sum(swe_d[i:ly_tot]))

        # dictionary of lists
        snowpack_d = pd.DataFrame({'h': h_d, "swe": swe_d, "swe_hat": swe_hat_d, "age": age_d})
        H_d = sum(snowpack_d['h'])

        a = snowpack_d.head(ly).apply(self.__compactH, axis=1, args=(ts, ))
        b = pd.DataFrame(np.zeros((ly_tot - ly, 4)))
        b.columns = ['h', "swe", "age", "rho"]

        self._snowpack_dd = pd.concat([a, b])
        # rownames(snowpack.dd.row) << - self.paste0("dd.layer", 1: nrow(snowpack.dd))
        return self._snowpack_dd

    def __drenchH(self, t, ly, ly_tot, day_tot, Hobs, h, swe, age, H, SWE, ts):
        Hobs_d = Hobs[t]
        h_d = h[:, t]
        swe_d = swe[:, t]
        age_d = age[:, t]

        self._current_day_info_string += "melt "

        runoff = 0
        # distribute mass top-down
        for i in reversed(range(ly)):
            if sum([element for j, element in enumerate(h_d) if j != i]) + swe_d[i] / self.rho_max - Hobs_d >= self.prec:
                # layers is densified to rho_max
                h_d[i] = swe_d[i] / self.rho_max
            else:
                # layer is densified as far as possible
                # but doesnt reach rho_max
                h_d[i] = swe_d[i] / self.rho_max + abs(
                    sum([element for j, element in enumerate(h_d) if j != i]) + swe_d[i] / self.rho_max - Hobs_d)
                break

        true_false_list = [self.rho_max - swe_d_val / h_d_val <= self.prec for swe_d_val, h_d_val in
                           zip(swe_d[:ly], h_d[:ly])]

        if all(true_false_list):
            self._current_day_info_string += "no further compaction "
            # produce runoff if sum(h_d) - Hobs_d is still > 0
            self._current_day_info_string += "runoff "
            # decrease swe from all layers?
            # or beginning with lowest?
            # swe_d[1:ly] <- swe_d[1:ly] - (sum(h_d) - Hobs_d) * rho_max
            scale = Hobs_d / sum(h_d)
            runoff = (sum(h_d) - Hobs_d) * self.rho_max  # excess is converted to runoff [kg/m2]
            h_d = h_d * scale  # all layers are compressed (and have rho_max) [m]
            swe_d = swe_d * scale
            # self._current_day_info_string += str(runoff)

        else:
            self._current_day_info_string += "compaction "

        h[:, t] = h_d
        swe[:, t] = swe_d
        age[:, t] = age_d
        H[t] = sum(h[:, t])
        SWE[t] = sum(swe[:, t])
        #
        # no further compaction possible
        # snowpack_tomorrow <- cbind(h = h_d, swe = swe_d, age = age_d, rho = swe_d/h_d)
        # colnames(snowpack_tomorrow) <- c("h","swe","age","rho")

        snowpack_tomorrow = self.__dry_metamorphism(h[:, t], swe[:, t], age[:, t], ly_tot, ly, ts)

        # set values for next day
        rl = self.__assignH(snowpack_tomorrow, h, swe, age, H, SWE, t, day_tot)
        h = rl["h"]
        swe = rl["swe"]
        age = rl["age"]
        H = rl["H"]
        SWE = rl["SWE"]

        return {'h': h, "swe": swe, "age": age, 'H': H, "SWE": SWE}


if __name__ == "__main__":
    path_to_hsdata = "sample_hsdata.csv"
    snow_to_swe = SnowToSwe()
    swe_pandas_df = snow_to_swe.convert_csv(path_to_hsdata, path_to_output_csv="sample_out.csv", verbose=True)
    hs_data_as_list = pd.read_csv(path_to_hsdata)["hs"].tolist()
    swe_list = snow_to_swe.convert_list(hs_data_as_list, 24, verbose=True)

    # results match with given R model example of Winkler et al. 2020 (git repo)
    print("Mean", np.mean(swe_list))
    print("Max", max(swe_list))
    print("Sum", sum(swe_list))
