# AET：auxiliary and embedded teacher mutual knowledge distillation for classification

Abstract: Online Knowledge Distillation (OKD) has emerged as a powerful technique for model compression, eliminating the need for pre-trained teachers in traditional methods. While recent advancements in feature fusion have further improved OKD's capabilities, existing approaches solely focus on final-layer fusion, potentially hindering the effectiveness of the fused classifier. In this work, we propose a novel Auxiliary and Embedded Teacher (AET) approach to tackle these challenges. AET addresses the critical issues of feature fusion position selection and potential performance degradation after fusion. We introduce embedded teachers, formed by combining multiple mid-level sub-networks, to promote mutual learning among student networks. Additionally, auxiliary teachers provide enriched information and guide the fusion classifier, ultimately enhancing overall performance. Extensive evaluations on four benchmark datasets (CIFAR-10/100, CINIC-10, and ImageNet2012) demonstrate the superiority of the proposed AET approach. Code is available: https://github.com/JSJ515-Group/AET

![image](https://github.com/JSJ515-Group/AET/assets/113502037/fc9ca204-2c81-44d3-a6e6-63986c2548d1)
![image](https://github.com/JSJ515-Group/AET/assets/113502037/326bd806-5763-465a-abee-8d2a40218361)



