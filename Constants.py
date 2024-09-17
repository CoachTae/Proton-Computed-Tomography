# PPF = Protons Per Frame
median_ppf = [2.848585498,  # 80 MeV
              3.19390456,    # 100 MeV
              3.24607426,    # 120 MeV
              3.84116720,    # 140 MeV
              3.131233897,  # 160 MeV
              4.94866067,    # 180 MeV
              6.110669848,  # 200 MeV
              5.751071088]  # 220 MeV

mean_ppf = [2.997789775,    # 80 MeV
            3.171535594,    # 100 MeV
            3.29144673,      # 120 MeV
            3.837730416,    # 140 MeV
            3.137407155,    # 160 MeV
            5.010778488,    # 180 MeV
            6.208131012,    # 200 MeV
            5.621536485]    # 220 MeV

real_energies = [80.3,
                 100.7,
                 119.7,
                 140.2,
                 159.9,
                 179.5,
                 200.4,
                 221.3]

# MEDIAN RUNS
med80 = "SC00100.ARW"
med100 = "SC00161.ARW"
med120 = "SC00183.ARW"
med140 = "SC00247.ARW"
med160 = "SC00071.ARW"
med180 = "SC00266.ARW"
med200 = "SC00318.ARW"
med220 = "SC00372.ARW"
med5cm = "PH00012.ARW"
med10cm = "PH00074.ARW"
med15cm = "PH00107.ARW"
medmuscle = "MB00000.ARW"
medbone = "MB00040.ARW"

median_runs = [med80,
               med100,
               med120,
               med140,
               med160,
               med180,
               med200,
               med220,
               med5cm,
               med10cm,
               med15cm,
               medmuscle,
               medbone]

# MEAN RUNS
mean80 = med80
mean100 = med100
mean120 = "SC00198.ARW"
mean140 = "SC00248.ARW"
mean160 = "SC00052.ARW"
mean180 = "SC00293.ARW"
mean200 = med200
mean220 = "SC00352.ARW"
mean5cm = "PH00037.ARW"
mean10cm = med10cm
mean15cm = "PH00105.ARW"
meanmuscle = medmuscle
meanbone = "MB00040.ARW"

mean_runs = [mean80,
             mean100,
             mean120,
             mean140,
             mean160,
             mean180,
             mean200,
             mean220,
             mean5cm,
             mean10cm,
             mean15cm,
             meanmuscle,
             meanbone]

regular_runs = ['80 MeV', '100 MeV', '120 MeV', '140 MeV', '160 MeV', '180 MeV', '200 MeV', '220 MeV']

tissue_runs = ['Muscle', 'Bone']

acrylic_runs = ['2 Blocks', '4 Blocks', '6 Blocks']

# Used for naming graphs for the acrylic runs
acrylic_runs_cm = ['5cm Acrylic', '10cm Acrylic', '15cm Acrylic']
