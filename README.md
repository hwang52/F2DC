# Domain-Skewed Federated Learning with Feature Decoupling and Calibration

This is an official implementation of the following paper:
> Huan Wang, Jun Shen, Jun Yan, Guansong Pang. *"Domain-Skewed Federated Learning with Feature Decoupling and Calibration"*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), CVPR 2026.
---

**Abstract:** Federated learning (FL) allows distributed clients to collaboratively train a global model in a privacy-preserving manner. However, one major challenge is domain skew, where clients' data originating from diverse domains may hinder the aggregated global model from learning a consistent representation space, resulting in poor generalizable ability in multiple domains. In this paper, we argue that the domain skew is reflected in the domain-specific biased features of each client, causing the local model's representations to collapse into a narrow low-dimensional subspace. We then propose **F**ederated **F**eature **D**ecoupling and **C**alibration (**F2DC**), which liberates valuable class-relevant information by calibrating the domain-specific biased features, enabling more consistent representations across domains. A novel component, Domain Feature Decoupler (DFD), is first introduced in F2DC to determine the robustness of each feature unit, thereby separating the local features into domain-robust features and domain-related features. A Domain Feature Corrector (DFC) is further proposed to calibrate these domain-related features by explicitly linking discriminative signals, capturing additional class-relevant clues that complement the domain-robust features. Finally, a domain-aware aggregation of the local models is performed to promote consensus among clients. Empirical results on three popular multi-domain datasets demonstrate the effectiveness of the proposed F2DC and the contributions of its two modules.

---

Here is an example to run F2DC on the PACS dataset:


```python
python3 main_F2DC.py --model f2dc --parti_num 10 --dataset fl_pacs --device_id 0
```