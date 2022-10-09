# Distributed learning system with local average regression

This is a novel distributed learning scheme. It authorises local agents the autonomy to determine their learning algorithms, algorithmic parameters, and attendance, and develops a corresponding qualification mechanism to ensure high-quality synthesized results.

Taking collaborative diagnosis as an example, this scheme adopts local average regression, a nonparametric paradigm, as the local processing algorithm, since this algorithm obeys some basic principles of clinical treatment to mimic the treatment process.




## Contents

- [Prerequisite](#Prerequisite)
- [Usage](#Usage)
- [Citation](#Citation)
- [Contacts](#Contacts)
- [License](#license)

## Prerequisite

Python 3.7

## Usage

Considerations when running the code:

1. Set the folder at first: right click on the folder Distibuted learning system with local average regression --> Mark Directory as --> Sources Root.

   If different folders call function.py with the same name, such as function_same_block.py, call confusion will occur. Therefore, remember to cancel the 'Sources Root' of the temporarily unused folder (by right click on the current folder --> Mark Directory as --> Unmark as Sources Root).

2. Select [Function_same_block.py](Function_same_block.py)  or [Function_diff_block.py](Function_diff_block.py)  depending on the data sizes owned by local agents when runing each .py in the file [Algorithm](Algorithm) and  [Parameter_training](Parameter_training). Specifically, if the amount of data held by the local agents is the same (equal-sized setting), select Function_same_block.py, otherwise, select the Function_diff_block.py (unequal-sized setting).

3. Adjust the specific algorithm NWK(gaussian),NWK(naive), NWK(Epanechnikov), corresponding to GE_gaussiank, GE_naivek, GE_Epank in [Function_same_block.py](Function_same_block.py)  and [Function_diff_block.py](Function_diff_block.py)  when training localization parameters by each .py in the file [Parameter_training](Parameter_training).

   ```sh
   error_validation = GE_gaussiank(x_train, y_train, x_test, y_test, h, d, s)   
   # Adjust the specific algorithm GE_gaussiank, GE_naivek, GE_Epank
   ```

4. Download of real data: 

   * **Insurance data:**

     https://www.kaggle.com/mirichoi0218/insurance 

   * **Warfarin dose data:**

      https://www.pharmgkb.org/downloads

     find: From Related Projects-- International Warfarin Pharmacogenetics Consortium (IWPC) --IWPC Data Set

## Citation

If you use this code, please consider citing:

```
@article{liu2022enabling,
  title={Enabling Collaborative Diagnosis through Novel Distributed Learning System with Autonomy},
  author={Liu, Xiaotong and Wang, Yao and Tang, Shaojie and Lin, Shao-Bo},
  journal={Available at SSRN 4128032},
  year={2022}
}
```

## Contacts

Please open an issue for any questions or suggestions.

Thanks! 


## License

Code: under MIT license.
