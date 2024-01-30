
# DeepWind

Bin Wang and Junrui Shi and Binyu Tan and Minbo Ma and Feng Hong and Yanwei Yu and Tianrui Li, DeepWind: a heterogeneous spatio-temporal model for wind forecasting, Knowledge-Based Systems 286 (2024) 111385.



Constructed a heterogeneous model named DeepWind, leveraging the advantages of recursive and non-recursive branches, and designed a difference loss function to guide model training for capturing wind trends.

Proposed an effective transformation of the target variable based on meteorological domain knowledge and introduced a time-related embedding technique termed step embedding to enhance forecasting performance.

Identified the overlooked issue of evaluation inconsistency in time series forecasting and advocated the use of diverse metrics for comprehensive evaluation. Extensive experiments on real-world wind datasets demonstrate that the proposed DeepWind outperforms strong DL baselines on multiple metrics.

## Paradox: evaluation inconsistency

We found an inconsistency in the evaluation of the current study. As shown in Figure 1, the green line represents the ground truth, while the red and blue lines represent two different prediction results. Solely based on regression metrics, the blue line outperforms the red line by far. However, if we divide the predicted values into 5 shaded intervals, each corresponding to a category, and evaluate from a classification perspective, the red line's classification prediction aligns perfectly with the ground truth, while the accuracy of the blue line is 0. A model that performs exceptionally well in regression tasks may not necessarily excel in classification tasks. Nevertheless, in many regression problems, classification evaluation is equally important, especially for disaster warnings such as for strong winds, high waves, and floods.

| <img src="F:/ouc/KBS/%E5%9B%BE%E8%A1%A8/reg_class_compare_00.png" alt="img" style="zoom: 30%;" /> |
| :----------------------------------------------------------: |
| Figure 1: An extreme example to illustrate the evaluation inconsistency. Note that forecast 2 exhibits superior performance (MSE: 1.05, MAE: 0.94) compared to forecast 1 (MSE: 4.16, MAE: 1.96) on regression metrics. However, if wind speeds are categorized into 5 classes, forecast 1 achieves 100\% accuracy while forecast 2 dwindles to zero accuracy. |


## Transformation of wind UV components

Particularly, following Equation (2) and Equation (3), the U and V at 10 meters, which is denoted by $(U_{10m}, V_{10m})$,  will be transformed as $(U_{10m}, V_{10m}) \rightarrow (wdir_{10m},spd_{10m}) \rightarrow (sin_{10m}, cos_{10m},spd_{10m})$ in turn.

Equation (2):
$$ {Equation (2)}
sin=\sin (wdir  /180^{\circ} \times \pi ) 
\\
cos=\cos (wdir  /180^{\circ} \times \pi )
$$
Equation (3)
$$ {Equation (3)}
wdir_{10m} = 270^{\circ}  -180^{\circ}*arctan2(V_{10m},U_{10m})/\pi
\\
spd_{10m} = \sqrt{U_{10m}^{2}   +V_{10m}^{2}}
$$

| <img src="F:/ouc/KBS/%E5%9B%BE%E8%A1%A8/wdir_00(1).png" alt="img" style="zoom: 25%;" /> |
| :----------------------------------------------------------: |
| Figure 2: Conversion relationship among $U$ component, $V$ component, wind speed, and wind direction. $wdir_{10m}$ represents the direction at 10 meters, $\theta$ represents the degree of wind direction; $spd_{10m}$ represents the wind speed at 10 meters, $U_{10m}$ and $V_{10m}$ represent the decomposed wind speeds, calculated via the equations in this figure. |

## Heterogeneous model architecture

|    ![img](F:/ouc/KBS/%E5%9B%BE%E8%A1%A8/DeepWind_00.png)    |
| :---------------------------------------------------------: |
| Figure 3: The entire architecture of the proposed DeepWind. |

Most deep learning models are homogeneous, either a purely recursive mechanism such as RNN or a purely non-recursive structure such as CNN. CNN-type models are good at extracting spatial dependencies in input data. As an efficient variant of RNN models, GRU specializes in capturing sequential dependencies within the data \cite{DBLP:journals/access/GangQiangBRY23, DBLP:journals/cea/WangWTTLQ23}. 

### Branch fusion

$$ {Equation (4)}
W_{GRU} = \text{Softplus} ( initW_{GRU} ) 
\\
W_{CNN} = \text{Softplus} ( initW_{CNN} ) 
\\
\hat{Y} = W_{GRU} \times \hat{Y}_{GRU} + W_{CNN} \times \hat{Y}_{CNN}
$$



###Trend difference loss

we proposed a trend difference loss to stabilize the trend learning between two adjacent forecasting steps $t$ and $t-1$, which was shown in Equation (5).

Equation (5): 
$$ {Equation (5)}
loss_{diff} = \vert \frac{1}{B \! \times \! (T\!-\!1) \! \times \! (D\!-\!2)} \sum^B_{b=1} \sum^T_{t=2} \sum^D_{d=3} [(\hat{Y}^{[b,t,d]} - \hat{Y}^{[b,t-1,d]}) - (Y^{[b,t,d]} - Y^{[b,t-1,d]}) ] \vert
$$


## Main results

![image-20240130235032888](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20240130235032888.png)



## Get started

1. Install Python>=3.9, PyTorch 1.9.0.

   ```bash
   pip install -r requirements.txt
   ```

2. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the multivariate and univariate experiment results by running the following shell code separately:

   ```bash
   bash ./scripts/run.sh
   ```

## Citation

If you find this repo useful, please cite our paper.

```
@article{WANG2024111385,
    title = {DeepWind: a heterogeneous spatio-temporal model for wind forecasting},
    journal = {Knowledge-Based Systems},
    volume = {286},
    pages = {111385},
    year = {2024},
    author = {Bin Wang and Junrui Shi and Binyu Tan and Minbo Ma and Feng Hong and Yanwei Yu and Tianrui Li}
}
```

## Contact

If you have any questions or suggestions, feel free to contact:

- Bin Wang (wangbin9545@ouc.edu.cn)

or describe it in Issues.

## Acknowledgement

The authors thanked Kaihua Zhang for his enthusiastic discussions. This work was supported by the Fundamental Research Funds for the Central Universities (no. 202313038) and the National Natural Science Foundation of China (no. 62176243, no. 41976185).