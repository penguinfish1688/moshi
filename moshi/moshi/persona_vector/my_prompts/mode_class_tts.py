
def mode_class_dataset_tts(dataset_path, answer_time=10):
    """
    Generate TTS audio files for a mode class dataset.
    the example is at Full-Duplex-Bench/data/mode_class/ as dataset_path
    for each dataset_path/*/input.json there are two fields: complete_sentence and incomplete_sentence (you can check the format yourself)
    for each input.json, generate two audio files: complete_sentence.wav and incomplete_sentence.wav
    the answer time is the time that you append to COMPLETE_SENTENCE.WAV after tts synthesis. 10 mean 10 senconds of silence after the complete sentence, which is the time that the model has to answer the question after listening to the complete sentence.
    refactor the args so that --trait is trait tts and --mode-class is mode class tts, and they are mutually exclusive. and --answer-time flag.
    only run static check as I don't have gpu
    """