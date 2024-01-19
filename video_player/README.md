
# Spaceport WEB Video Player 

This repository contains some web pages under examples that have perspective and equirectangular projections with rendering using 4D Gaussian Splatting Repo. If you want to see those demo pages on your PC in order to have better switching speed between videos, do the following steps

Please run the following commands to run http server

```bash
pip install rangehttpserver
python -m RangeHTTPServer
```

Then download video files of perspective and equirectangular projections and uncompress them to the folder named 'media' (create it if does not exist) using the following [link](https://drive.google.com/drive/folders/1sQWfPb7Wn0ZMFkwqEsrObqPNYHN048OC?usp=sharing), and change 'basepath' variable defined on demo pages under 'examples' folder, giving the relative path of video files downloaded.

We have also some perspective demo pages living on our webpage [www.spaceport.tv](https://www.spaceport.tv)

https://spaceport.tv/demo/faruk_teknopark_13-12-2023_persp_cdn.htm 

https://spaceport.tv/demo/faruk_teknopark_03-01-2024_persp_cdn.htm