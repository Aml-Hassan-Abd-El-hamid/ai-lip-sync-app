# AI Lip Sync

The project is essentially a part of an interview process with the company named [DigiSay](https://digisay.breezy.hr/), I received a mail with the following task:

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
        
Evaluation Criteria:<br>
Candidates will be evaluated based on the following criteria:<br>
        &emsp;&emsp;&emsp;&emsp;● Model Performance: How well does the model synchronise lip movements<br>
        &emsp;&emsp;&emsp;&emsp;with audio?<br>
        &emsp;&emsp;&emsp;&emsp;● Code Quality: Evaluate the clarity, organisation, and readability of<br>
        &emsp;&emsp;&emsp;&emsp;the code. Ensure that the code is well-documented.<br>
        &emsp;&emsp;&emsp;&emsp;● Understanding of Machine Learning Concepts: Assess the candidate's<br>
        &emsp;&emsp;&emsp;&emsp; understanding of machine learning principles and their ability to<br>
        &emsp;&emsp;&emsp;&emsp;apply them to a practical problem.<br>
        
I was given about 96 hours to accomplish this task, I spent the first 12 hours sick with a very bad flu and no proper internet connection so now I have 84 hours, let's go!

Given the provided hint from the company, "You can refer to state of the art models in lip-syncing.", I started looking into the available open-source pre-trained model that can accomplish this task and most available resources pointed towards **Wav2Lip**. I found a couple of interesting tutorials for that model that I will add to this project later.

**Pushing the checkpoints files:**<br>

Given the size of those kind of files, I had to use git lfs, here's how to do it:<br>
1- Follow the installation instructions that are suitable for your system from [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) <br>
2- Use the command `git lfs track "*.pth"` to let git lfs know that those are your big files.<br>
3- When pushing from the command line -I usually use VS code but it usually doesn't work with big files like `.pth` files- you need to generate a personal access token, to do so, follow the instructions from [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token), and then copy the token<br>
4- When pushing the file from the terminal you will be asked to pass a password, don't pass your GitHub profile password, instead pass your personal access token that you got from step 3.

**Things I changed in the wav2lip and why:**<br>

In order to work with and deply the wav2lip model I had to do the following changes:<br>
1- Changed the `_build_mel_basis()` function in `audio.py`, I had to do that in order to be able to work with `librosa>=0.10.0` pakadge, check this [issue](https://github.com/Rudrabha/Wav2Lip/issues/550) for more details.<br>
2- Changed the `main()` function at the `inferance.py` to make it take an output from the `app.py` directly insted of using the command line arguments.<br>
3- Since I'm using streamlit for deployment and streamlit Cloud doesn't support GPU, I had to change the device to work with `cpu` insted of `cuda`.<br>
4- I did other minor changes like changing path to file or modify import statements.

**I broke down my work plan for that project into the following pieces:**<br>
- [x] 1- From a video & audio to a lip-synced video: a function that takes a video of the avatar talking + the audio and produces a lip-synced video using **Wav2Lip**<br>
- [ ] * Bounce: deploy the function from step 1 using Streamlit<br>
- [ ] 2- From an avatar image to a video: a function that takes an avatar image and animates and produces the video that should be passed to the function from step 1<br>
- [ ] * Bounce: add the function from step 2 to the deployed Streamlit website and take the audio from the user as with a record function.<br>
- [ ] * Bounce: let the model create its own avatar using a prompt.<br>
