# Small WEC Modeling Data Extractor and Visualizer

This repository contains a Python script designed to extract and
visualize data from the [Small WEC Performance Modeling
Tool](https://apps.openei.org/swec/devices). The script converts and
visualizes json data from the [OpenEI Small Scale WEC Performance
Modeling Data Submission 380](https://mhkdr.openei.org/submissions/380).

## Numerical Data

- Numerical data is stored as csv files in the
  `./data/b1_vap/<test_case>/` directory.:

  - By test case performance data as columnar data:
    `./data/b1_vap/<test_case>/<test_case>_columnar.csv`
  - By test case metadata:
    `./data/b1_vap/<test_case>/<test_case>_metadata.json`
  - By test case, by variable peak period numerical binned matrices:
    `./data/b1_vap/<test_case>/matricies/peak_period/<test_case>_<variable>.csv`
  - By test case, by variable energy period numerical binned matrices:
    `./data/b1_vap/<test_case>/matricies/energy_period/<test_case>_<variable>.csv`

## Visualizations

- Visualizations: `./viz/by_run/<test_case>/`
  - By test case, by variable peak period matrices as heatmap:
    `./viz/by_run/<test_case>/matricies/peak_period/<test_case>_<variable>.png`
  - By test case, by variable energy period matrices as heatmap:
    `./viz/by_run/<test_case>/matricies/energy_period/<test_case>_<variable>.png`
  - By variable, by test case peak period matrices as heatmap:
    `./viz/by_col/<test_case>/matricies/peak_period/<test_case>_<variable>.png`
  - By variable, by test case energy period matrices as heatmap:
    `./viz/by_col/<test_case>/matricies/energy_period/<test_case>_<variable>.png`

![McCabe Wave Pump - 10m Scale - Power
Average](./viz/by_col/power_average/peak_period/10m_scale_mccabe_power_average_peak_period_matrix.png)

For data details please refer to the [Small Scale WEC Performance
Modeling Data](https://mhkdr.openei.org/submissions/380) page.

## References

King, Tom, Jim McNally, Nicole Taverna, RJ Scavo, Scott Jenne, and Jon
Weers. 2021. “Small Scale WEC Performance Modeling Data.” Marine and
Hydrokinetic Data Repository, National Renewable Energy Laboratory,
https://doi.org/10.15473/1838617. <https://doi.org/10.15473/1838617>.
