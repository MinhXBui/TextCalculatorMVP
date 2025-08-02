It's a pivot table, but for text data. That's about it, I guess. Cause after all: "The application capability is being limited by my creativity."

To run it locally:
1. pip install -r requirements.txt
2. In the .streamlit folder, add secrets.toml and then add: GOOGLE_API_KEY="YourGeminiKey"
3. Hit: "streamlit run main.py" to see it on your browser

To adjust the file rows limit and file size limit:
1. Filesize: .streamlit/config.toml --> maxUploadSize =  to filesize limit you want in MB
2. RowLimit: main.py line 60, you can change the value from 5000 to your preferred value.

You can also change it out for a more capable Gemini Model, but be careful of the cost if using Gemini 2.5 Pro.

Here is a quick demo/tutorial by me on YouTube on how to use the software:
[https://www.youtube.com/watch?v=iJKgxyn8RPU&embeds_referring_euri=http%3A%2F%2Flocalhost%3A8501%2F&source_ve_path=OTY3MTQ](https://www.youtube.com/watch?v=iJKgxyn8RPU&ab_channel=TextCalculator)

Check it out and let me know what you think. It would be really interesting to see an alternative application besides analyzing marketing survey open responses and customer reviews.


