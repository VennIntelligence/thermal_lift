# 论文引用清单（供生成 refs.bib 用）

> 从 `docs/paper/zh/` 各章节中提取的所有 `\citep`/`\citet` 引用 key。
> 每条给出 key、推测的完整引文信息、出处章节。
> 请用 Codex / GPT Pro 根据本清单生成 `docs/paper/aaai/refs.bib`。
>
> 当前唯一引用来源：§1 引言 + §2 相关工作（§3–§8 尚未插入引用）。
> 总计 **32 条**唯一引用（含 2 条新增候选）。

---

## §1 引言 + §2 相关工作 共用引用

### 经典多帧超分辨

1. **tsai1984multiframe**
   Tsai, R. Y. & Huang, T. S. (1984). Multiframe image restoration and registration. *Advances in Computer Vision and Image Processing*, 1, 317–339.

2. **irani1991improving**
   Irani, M. & Peleg, S. (1991). Improving resolution by image registration. *CVGIP: Graphical Models and Image Processing*, 53(3), 231–239.

3. **farsiu2004fast**
   Farsiu, S., Robinson, M. D., Elad, M., & Milanfar, P. (2004). Fast and robust multiframe super resolution. *IEEE Transactions on Image Processing*, 13(10), 1327–1344.

4. **fruchter2002drizzle**
   Fruchter, A. S. & Hook, R. N. (2002). Drizzle: A method for the linear reconstruction of undersampled images. *PASP*, 114(792), 144–152.

5. **rudin1992nonlinear**
   Rudin, L. I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation based noise removal algorithms. *Physica D*, 60(1–4), 259–268.

6. **bredies2010tgv**
   Bredies, K., Kunisch, K., & Pock, T. (2010). Total generalized variation. *SIAM Journal on Imaging Sciences*, 3(3), 492–526.

7. **elad1997restoration**
   Elad, M. & Feuer, A. (1997). Restoration of a single superresolution image from several blurred, noisy, and undersampled measured images. *IEEE Transactions on Image Processing*, 6(12), 1646–1658.

8. **hardie1997joint**
   Hardie, R. C., Barnard, K. J., & Armstrong, E. E. (1997). Joint MAP registration and high-resolution image estimation using a sequence of undersampled images. *IEEE Transactions on Image Processing*, 6(12), 1621–1633.

### 深度 burst 超分辨

9. **bhat2021deep**
   Bhat, G., Danelljan, M., Van Gool, L., & Timofte, R. (2021). Deep burst super-resolution. *CVPR*.

10. **bhat2021ntire**
    Bhat, G., Danelljan, M., & Timofte, R. (2021). NTIRE 2021 challenge on burst super-resolution: Methods and results. *CVPR Workshops*.

11. **dudhane2022burst**
    Dudhane, A., Zamir, S. W., Khan, S., Khan, F. S., & Yang, M.-H. (2022). Burst image restoration and enhancement. *CVPR*. (BIPNet)

12. **dudhane2023burstormer**
    Dudhane, A., Zamir, S. W., Khan, S., Khan, F. S., & Yang, M.-H. (2023). Burstormer: Burst image restoration and enhancement transformer. *CVPR*.

13. **deudon2020highres**
    Deudon, M., Kalaitzis, A., Goytom, I., et al. (2020). HighRes-net: Recursive fusion for multi-frame super-resolution of satellite imagery. *arXiv:2002.06460*.

14. **salvetti2020rams**
    Salvetti, F., Mazzia, V., Khaliq, A., & Chiaberge, M. (2020). Multi-image super resolution of remotely sensed images using residual attention deep neural networks. *Remote Sensing*, 12(14), 2207.

### 热成像超分辨

15. **rivadeneira2020thermal**
    Rivadeneira, R. E., Suárez, P. L., Sappa, A. D., & Vintimilla, B. X. (2020). Thermal image super-resolution challenge — PBVS 2020. *CVPR Workshops*.

16. **rivadeneira2023thermal**
    Rivadeneira, R. E., et al. (2023). Thermal image super-resolution challenge results — PBVS 2023. *CVPR Workshops*.

17. **chudasama2020therisurnet**
    Chudasama, V., et al. (2020). TherISuRNet: A computationally efficient thermal image super-resolution network. *CVPR Workshops*.

18. **rivadeneira2020novel**
    Rivadeneira, R. E., et al. (2020). A novel architecture for thermal image super-resolution. *CVPR Workshops* (PBVS).

### 合成到真实的训练与退化建模

19. **bellkligler2019blind**
    Bell-Kligler, S., Shocher, A., & Irani, M. (2019). Blind super-resolution kernel estimation using an internal-GAN. *NeurIPS*.

20. **wang2021real**
    Wang, X., Xie, L., Dong, C., & Shan, Y. (2021). Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data. *ICCV Workshops*.

21. **zhang2021designing**
    Zhang, K., Liang, J., Van Gool, L., & Timofte, R. (2021). Designing a practical degradation model for deep blind image super-resolution. *ICCV*.

### 无 GT 评估

22. **vanheel2005fourier**
    van Heel, M. & Schatz, M. (2005). Fourier shell correlation threshold criteria. *Journal of Structural Biology*, 151(3), 250–262.

23. **nieuwenhuizen2013measuring**
    Nieuwenhuizen, R. P. J., et al. (2013). Measuring image resolution in optical nanoscopy. *Nature Methods*, 10(6), 557–562.

24. **banterle2013fourier**
    Banterle, N., Bui, K. H., Lemke, E. A., & Beck, M. (2013). Fourier ring correlation as a resolution criterion for super-resolution microscopy. *Journal of Structural Biology*, 183(3), 363–367.

25. **mittal2013making**
    Mittal, A., Soundararajan, R., & Bovik, A. C. (2013). Making a "completely blind" image quality analyzer. *IEEE Signal Processing Letters*, 20(3), 209–212. (NIQE)

### 逆问题中的零空间与数据一致性

26. **schwab2019deep**
    Schwab, J., Antholzer, S., & Haltmeier, M. (2019). Deep null space learning for inverse problems: Convergence analysis and rates. *Inverse Problems*, 35(2).

27. **chen2021equivariant**
    Chen, D. & Davies, M. E. (2021). Equivariant imaging: Learning beyond the range space. *ICCV*.

28. **ulyanov2018deep**
    Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2018). Deep image prior. *CVPR*.

29. **heckel2019deep**
    Heckel, R. & Hand, P. (2019). Deep decoder: Concise image representations from untrained non-convolutional networks. *ICLR*.

30. **schlemper2018deep**
    Schlemper, J., Caballero, J., Hajnal, J. V., Price, A. N., & Rueckert, D. (2018). A deep cascade of convolutional neural networks for dynamic MR image reconstruction. *IEEE TMI*, 37(2), 491–503.

---

## 新增候选引用（§1 首句 thermal NDT）

> 从以下两条中选一或同时使用，替换 `TODO_thermal_ndt`。
> 推荐 **breitenstein2010lockin**（直接面向半导体器件热成像检测）。

31. **breitenstein2010lockin**
    Breitenstein, O., Warta, W., & Langenkamp, M. (2010). *Lock-in Thermography: Basics and Use for Evaluating Electronic Devices and Materials* (2nd ed.). Springer Series in Advanced Microelectronics, Vol. 10. Springer, Berlin/Heidelberg.
    — 半导体器件热成像检测的标准专著，涵盖 lock-in 技术、缺陷定位、空间分辨率极限。

32. **maldague2001infrared**
    Maldague, X. P. V. (2001). *Theory and Practice of Infrared Technology for Nondestructive Testing*. Wiley, New York.
    — 红外无损检测领域的经典教材，覆盖被动/主动热成像原理与工业案例。

---

## §4 方法新增引用

33. **beck2009fast**
    Beck, A. & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems.

34. **ronneberger2015unet**
    Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation.

35. **wu2018group**
    Wu, Y. & He, K. (2018). Group normalization.

36. **hu2018squeeze**
    Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks.

37. **shi2016realtime**
    Shi, W., Caballero, J., Huszár, F., Totz, J., et al. (2016). Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network.

38. **aitken2017checkerboard**
    Aitken, A., Ledig, C., Theis, L., Caballero, J., Wang, Z., & Shi, W. (2017). Checkerboard artifact free sub-pixel convolution.

39. **chambolle2011firstorder**
    Chambolle, A. & Pock, T. (2011). A first-order primal-dual algorithm for convex problems with applications to imaging.

40. **wang2004image**
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity.

---

## 统计

| 范围 | 唯一引用数 |
|------|-----------|
| §1 引言 | 8（含 1 个 TODO） |
| §2 相关工作 | 28（与 §1 有大量重叠） |
| §1 + §2 去重 | 30 |
| 新增候选（§1） | 2 |
| §4 方法新增 | 8 |
| **合计** | **40** |

§5–§8 尚未插入引用，预计最终 42–48 条。对 AAAI/CVPR 长文属正常范围。
