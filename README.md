# Blooming
The first render pass generates 4 results, an intermediate image that stores the brightest part in a PBR scene, 
an intermediate image texture that stores the vertical Gaussian effect, an intermediate image texture that stores the horizontal Gaussian effect, and an intermediate image texture that stores the actual PBR scene. 
The second render pass is used to achieve full-screen post-processing that combines the blooming result and the actual PBR scene.

In terms of Gaussian weights, we use one-dimension Gaussian formula to calculate weights for neighboring pixels in the vertical and horizontal directions:

![image](https://github.com/iTzzYiuShaw/Render-To-Texture-and-Blooming/assets/110170509/d960379d-5ad0-498f-9722-872fc2ac303d)

Result screenshot without glooming

![image](https://github.com/iTzzYiuShaw/Render-To-Texture-and-Blooming/assets/110170509/3fe2331f-4545-4b7a-b46e-72b81860cac9)

Result screenshot with glooming

![image](https://github.com/iTzzYiuShaw/Render-To-Texture-and-Blooming/assets/110170509/68f86a74-ded7-4cce-b286-53fc7ccd9c18)


