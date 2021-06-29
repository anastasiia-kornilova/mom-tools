# mom-tools



MOM (Mutually Orthogonal Metric) is a metric that evaluates trajectory quality via estimation inconsistency of the map aggregated from registered point clouds via this trajectory. It uses statistics obtained from points from mutually orthogonal surfaces and, therefore, strongly correlates with Relative Pose Error. This repository provides source code for this metric and examples of applications on widespread LiDAR datasets.



For more details about method and experiments, please, refer to our preprint: "Be your own Benchmark: No-Reference Trajectory Metric on Registered Point Clouds", Anastasiia Kornilova, Gonzalo Ferrer, 2021 [[arxiv]](https://arxiv.org/abs/2106.11351).



The image below demonstrates an example of applying map metric to compare GT poses and poses from ICP algorithm on KITTI dataset via evaluating small-scale map inconsistency.  As a result, MOM demonstrates superiority on small-scale mapping in comparison to GT poses that could be noisy on small-scale distances.

![](fig/teaser.png)

### Citing

If you use this tool for your research, please, cite:

```latex
@misc{kornilova2021benchmark,
      title={Be your own Benchmark: No-Reference Trajectory Metric on Registered Point Clouds}, 
      author={Anastasiia Kornilova and Gonzalo Ferrer},
      year={2021},
      eprint={2106.11351},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
