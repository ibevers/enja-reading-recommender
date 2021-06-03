from google.cloud import translate
import os


def translate_text(text="名前はアイザックです", project_id="[removed for
                   security reasons]"):
    """Translating Text."""

    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": "ja",
            "target_language_code": "en-US",
        }
    )

    # Display the translation for each input text provided
    for translation in response.translations:
        print("Translated text: {}".format(translation.translated_text))
        return "{}".format(translation.translated_text)

if __name__ == "__main__":

    #src = "/Users/isaacbevers/CS229_Final_Project/test-texts/"
    #dest = "/Users/isaacbevers/CS229_Final_Project/translated-texts/"
    #filename = "shinsaku.txt"
    #text = ''
    #with open(src + filename, 'r') as file:
    #    text = file.read()
    #dest_file = open(dest + filename, 'x')
    #translated_text = translate_text(text=text)
    #dest_file.write(translated_text)
    #dest_file.close()
    src = "/Users/isaacbevers/CS229_Final_Project/small-sample"
    dest = "/Users/isaacbevers/CS229_Final_Project/translated-texts/"
    for filename in os.listdir(src):
        if os.path.isfile(dest + filename)==False:
            text = ''
            with open(src + "/" + filename, 'r', encoding="utf-8") as file:
                text = file.read()
            dest_file = open(dest + filename, 'x')
            translated_text = translate_text(text=text)
            dest_file.write(translated_text)
            dest_file.close()
