import gpt_2_simple as gpt2
from deltaPrediction import *
import pandas as pd


sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

# pre = '<|newsubmission|>\n\nCMV: The internet is a "monopoly" where the price of knowledge is controlled by only a few, and the popularity of existing platforms and services is driven by a desire to sell more.\nI think that a lot of the criticism of the internet as a marketplace and a place where everyone is competing for resources is a bit unfair.\n<|newcomment|>'

pre = """

CMV: Seeing a person dressed as a woman that is clearly a man will never not be weird.
I know gender dysphoria is real and transgender people exist. I know that the world would be better off accepting that fact.

But in the push to accept transgender people in society, there seems to be this awkward game of pretend that is expected to be played.

It is one thing to simply adjust the pronouns you use to make them comfortable, but it is another thing entirely to be expected to convince yourself that they are the gender in which they identify. If you meet a trans woman that is clearly trans (most notably ones that used to be men with clear male facial features, endomorph or even receding hairlines) you are expected to actually believe she is a woman on a biological level.

There are some very lucky biological men that can transition and be incredibly convincing, but most aren't, and if you aren't one of those lucky people, you will never be seen as an actual woman no matter how much pretend we are expected to play. We can all act like they are women, but the brain already sorts them out as "men dressed as women" way before any conscious decision is made, and it will always be weird and awkward seeing these individuals no matter how nice and accepting we are about it.

<|newcomment|>
"""

iter = 50
comments = []
for i in range(iter):
    single_text = gpt2.generate(sess, prefix=pre, truncate="<|newsubmission|>", return_as_list=True)[0]
    # print("<|newsubmission|>\n")
    # print(single_text)
    gen_text = single_text.split("<|newcomment|>")[1]
    features = getFeatures(gen_text)
    prediction = getPrediction([features])[0][1]
    comments.append([gen_text, prediction])
    print(prediction)
    if prediction > 90:
        break

df = pd.DataFrame(comments)
df.to_csv('comments_and_preds.csv', index = None, header=False)
