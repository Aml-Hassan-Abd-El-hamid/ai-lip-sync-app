# AI Lip Sync

The project is essentially a part of an interview process with the company named [DigiSay](https://digisay.breezy.hr/), I received an email with the following task:

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

In order to work with and deploy the wav2lip model I had to make the following changes:<br>
1- Changed the `_build_mel_basis()` function in `audio.py`, I had to do that in order to be able to work with `librosa>=0.10.0` package, check this [issue](https://github.com/Rudrabha/Wav2Lip/issues/550) for more details.<br>
2- Changed the `main()` function at the `inferance.py` to make it take an output from the `app.py` directly instead of using the command line arguments.<br>
3- I took the `load_model(path)` function and added it to `app.py` and added `@st.cache_data` in order to only load the model once, instead of using it multiple times.<br>
4- Deleted the unnecessary files like the unused checkpoints.<br>
5- Since I'm using Streamlit for deployment and Streamlit Cloud doesn't support GPU, I had to change the device to work with `cpu` instead of `cuda`.<br>
6- I made other minor changes like changing the path to a file or modifying import statements.

**How to run the application locally:**<br>

1- clone the repo to your local machine.<br>
2- install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) if you don't have it already, and create a new virtual environment for the project, you can use another way to create your own environment but don't use [Poetry](https://python-poetry.org/) cause some of the modules needed to run the Wav2Lip is old and not compatible with poetry and you'll end up unable to install the requirement modules for this project.
3- open your terminal inside the project folder and run the following command: `pip install -r requirements.txt` to install the needed modules.<br>
4- open your terminal inside the project folder and run the following command: `streamlit run app.py` to run the streamlit application.<br>

**Video preview of the application:**<br>

**fast animation version**


https://github.com/Aml-Hassan-Abd-El-hamid/AI-Lip-Sync/assets/66205928/36577ccb-5ec6-4bb4-b7ff-44bb52a4f984

