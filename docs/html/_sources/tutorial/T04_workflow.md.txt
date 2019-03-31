### Example workflow

The example for shape symmetrization of band mapping data of `$\text{WSe}_2$`

1. Pick the source image (with distortion) from measured band mapping data, `$I(k_x, k_y, E)​$`, that contains salient features such as discernible intensity local maxima/minima. For `$\text{WSe}_2​$` data, the source image, `$I(k_x, k_y)​$`, is selected from the energy region near the valence band maxima `$(E \sim E_{\text{VBM}})​$`, where the sixfold symmetric pattern is visible and sharp.
2. Determine the pixel coordinates of the landmarks from the source image by manual selection. The seven landmarks consist of the three `$K$` and `$K’$` points on the outer edge and the `$\Gamma$` point at the center of the source image. These pixel coordinates form the reference point set `$\tilde{P}$`.
3. With manual landmark selection, the ordering and specification (i.e. center or vertex) of the points may be determine by the operator directly, but for algorithmically detected point set, subsequent steps are needed to separate the center (`$\Gamma$` point) from the vertices (`$K$` and `$K’$` points) and to order the vertices in a clockwise or counterclockwise fashion with respect to the center.
4. Calculate the geometric quantities to be used in the subsequent registration. This includes the average center-vertex distance, `$\overline{\Vert \tilde{P}_i - \tilde{P}_C \Vert}$`, and the average nearest neighbor vertex-vertex distance, `$\overline{\Vert \tilde{P}_i - \tilde{P}_j \Vert}_{\text{NN}}$`.
5. Generate a set of points as the target point set using the existing corresponding coordinates from the source image. For `$\text{WSe}_2$` data, select one (e.g. the first) of the `$K$` or `$K’$` points determined in step (4), compute the pixel coordinates of the of perfectly symmetric hexagon vertices with respect to the fixed points.
6. Determine the coordinate transform `$\mathcal{L}$` computationally using the reference and target point sets.
7. If necessary, adjust the pose (i.e. rotation and position) of the pattern such that a diagonal of the hexagon is parallel with one of the image axes (`$x$` or `$y$`). Combine the additional transforms with the symmetrization transform determined in step (6).
8. Apply the final version of the transform to the stack of images along the energy axis.
