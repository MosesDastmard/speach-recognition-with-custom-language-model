from . import config
import re
def purify_text(text):
    regex = ",".join(list(config.CHAR_SET))
    regex = fr'[{regex}]'
    pure_text = "".join(re.findall(regex, text)).lower()
    pure_text = re.sub(pattern=config.IGNORE_PUNC, repl="", string=pure_text)
    return pure_text.strip()

def shrink(text):
    stride = config.STRIDE
    texts = []
    if len(text) >= config.MAX_CHAR:
        start = 0
        end = start + config.MAX_CHAR
        while end <= len(text):
            subtext = text[start:end]
            start += stride
            end += stride
            texts.append(subtext)
        subtext = text[start:]
        texts.append(subtext)
        return texts
    else:
        return [text]