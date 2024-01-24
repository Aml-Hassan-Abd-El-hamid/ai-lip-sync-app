# AI Lip Sync

![Screenshot 2024-01-22 at 03-03-09 app · Streamlit](https://github.com/Aml-Hassan-Abd-El-hamid/AI-Lip-Sync/assets/66205928/d35f379f-f1ca-46e5-a113-bfa3f3c4c2f9)


The project started as a part of an interview process with some company, I received an email with the following task:

Assignment Object:<br>
          &emsp;&emsp;&emsp;&emsp;Your task is to develop a lip-syncing model using machine learning
          techniques. It takes an input image and audio and then generates a video
          where the image appears to lip sync with the provided audio. You have to
          develop this task using python3.

Requirements:<br>
        &emsp;&emsp;&emsp;&emsp;● Avatar / Image : Get one AI-generated avatar, the avatar may be for a<br>
        &emsp;&emsp;&emsp;&emsp;man, woman, old man, old lady or a child. Ensure that the avatar is<br>
        &emsp;&emsp;&emsp;&emsp;created by artificial intelligence and does not represent real<br>
        &emsp;&emsp;&emsp;&emsp;individuals.<br>
        &emsp;&emsp;&emsp;&emsp;● Audio : Provide two distinct and clear audio recordings—one in Arabic<br>
        &emsp;&emsp;&emsp;&emsp;and the other in English. The duration of each audio clip should be<br>
        &emsp;&emsp;&emsp;&emsp;no less than 30 seconds and no more than 1 minute.<br>
        &emsp;&emsp;&emsp;&emsp;● Lip-sync model: Develop a lip-syncing model to synchronise the lip<br>
        &emsp;&emsp;&emsp;&emsp;movements of the chosen avatar with the provided audio. Ensure the<br>
        &emsp;&emsp;&emsp;&emsp;model demonstrates proficiency in accurately aligning lip motions<br>
        &emsp;&emsp;&emsp;&emsp;with the spoken words in both Arabic and English.<br>
        &emsp;&emsp;&emsp;&emsp;Hint : You can refer to state of the art models in lip-syncing.<br>
         
I was given about 96 hours to accomplish this task, I spent the first 12 hours sick with a very bad flu and no proper internet connection so I had 84 hours!<br>
After submitting the task on time, I took more time to deploy the project on Streamlight, as I thought it was a fun project and would be a nice addition to my CV:)  

Given the provided hint from the company, "You can refer to state-of-the-art models in lip-syncing.", I started looking into the available open-source pre-trained model that can accomplish this task and most available resources pointed towards **Wav2Lip**. I found a couple of interesting tutorials for that model that I will share below.

**Pushing the checkpoints files:**<br>

Given the size of those kind of files, There are 2 ways to handle that. 

At the start, I had to use git lfs, here's how to do it:<br>

1- Follow the installation instructions that are suitable for your system from [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) <br>
2- Use the command `git lfs track "*.pth"` to let git lfs know that those are your big files.<br>
3- When pushing from the command line -I usually use VS code but it usually doesn't work with big files like `.pth` files- you need to generate a personal access token, to do so, follow the instructions from [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token), and then copy the token<br>
4- When pushing the file from the terminal you will be asked to pass a password, don't pass your GitHub profile password, instead pass your personal access token that you got from step 3.

But then Streamlit wasn't capable of even pulling the repo! so I uploaded the model checkpoints and some other files to Google Drive, put them in a public folder, and then used a module called gdown to download those folders when needed! here’s a [link](https://github.com/wkentaro/gdown) to that gdown, it’s straightforward to use and install.

**Things I changed in the wav2lip and why:**<br>

In order to work with and deploy the wav2lip model I had to make the following changes:<br>
1- Changed the `_build_mel_basis()` function in `audio.py`, I had to do that to be able to work with `librosa>=0.10.0` package, check this [issue](https://github.com/Rudrabha/Wav2Lip/issues/550) for more details.<br>
2- Changed the `main()` function at the `inferance.py` to directly take an output from the `app.py` instead of using the command line arguments.<br>
3- I took the `load_model(path)` function and added it to `app.py` and added `@st.cache_data` in order to only load the model once, instead of using it multiple times, I also modified it<br>
4- Deleted the unnecessary files like the checkpoints to make the Streamlit website deployment easier.<br>
5- Since I'm using Streamlit for deployment and Streamlit Cloud doesn't support GPU, I had to change the device to work with `cpu` instead of `cuda`.<br>
6- I made other minor changes like changing the path to a file or modifying import statements.

**Issues I had with Streamlit, during the deployment:**

This part is a documentation for me, just in case, I need to face an issue in the future and also could be helpful for any poor soul who would have to work with Streamlit:

1- 
```
Error downloading object: wav2lip/checkpoints/wav2lip_gan.pth (ca9ab7b): Smudge error: Error downloading wav2lip/checkpoints/wav2lip_gan.pth (ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8): batch request: git@github.com: Permission denied (publickey).: exit status 255

Errors logged to /mount/src/ai-lip-sync/.git/lfs/logs/20240121T212252.496674
```
This essentially Streamlit telling you that it can't handle that big file, upload it to Google Drive, and then load it using Python code later, and no `git lfs` won't solve the problem :)<br> 
A ground rule that I learned here is: that the lighter you make your app, the better and faster it is to deploy it.<br>
I opened a topic with that issue on the Streamlit forum, right [here](https://discuss.streamlit.io/t/file-upload-fails-with-error-downloading-object-wav2lip-checkpoints-wav2lip-gan-pth-ca9ab7b/60261)<br>

2- Other issues that I faced a lot were dependency issues -lots of them- and that was mostly due to the fact that I depended on `pipreqs` to write down my `requirements.txt`, that `pipreqs` missed up my modules, it added unneeded ones and missed others, unfortunately, it took me some time to discover that and really slowed me down.

3- 
```
 ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```
I faced that problem during importing `cv2` -`openCv`- and the solution was to install `libgl1-mesa-dev` and some other packages using `apt`, you can't just add such packages to the `requirements.txt`, you need to create a file named `packages.txt` to do so.

4- Streamlit can't handle heavy processing, I discovered that when I tried to deploy the `slow animation` button to process video input alongside recording to get more accurate lip-syncing, the application failed directly when I used that button -and I tried to use it twice :)-, and that kinda make sense as Streamlit doesn't have a GPU or even a high ram space -I don't have a good GPU but I have about 64GB ram which was enough to run that function locally- and to solve that issue, I initiated another branch to contain the deployment version that doesn't have the `slow animation` button and used that branch for deployment while kept the main branch containing that button.

**How to run the application locally:**<br>

1- clone the repo to your local machine.<br>
2- open your terminal inside the project folder and run the following command: `pip install -r requirements.txt` to install the needed modules.<br>
3- open your terminal inside the project folder and run the following command: `streamlit run app.py` to run the streamlit application.<br>

**Video preview of the application:**<br>

**fast animation version**<br>
Notice how only the lips are moving.

https://github.com/Aml-Hassan-Abd-El-hamid/AI-Lip-Sync/assets/66205928/36577ccb-5ec6-4bb4-b7ff-44bb52a4f984

**slower animation version**<br>
Notice how the eye and the whole face are moving instead of only the lips.<br>

Unfortunately, Streamlit can't handle the computational power that the slower animation version requires and that's why I made it only available on the offline version, which means that you need to run the application locally to try that version.

https://github.com/Aml-Hassan-Abd-El-hamid/AI-Lip-Sync/assets/66205928/26740856-52e5-4fe7-868d-3b9341e97064

The only difference between the fast and slow versions of animation here is the fact that the fast version passes only a photo while the slow version passes a video instead.
