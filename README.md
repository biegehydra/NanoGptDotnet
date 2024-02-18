## NanoGptDotnet
Translated from python code written by Andrej Karpathy https://www.youtube.com/watch?v=kCc8FmEb1nY

### Running on CUDA
The project is currently setup to run on a CPU. To run on CUDA go to [Shared.csporj](https://github.com/biegehydra/NanoGptDotnet/blob/master/src/Shared/Shared.csproj) and uncomments the line for the CUDA package and delete the line for the cpu package.

### Contributing

Feel free to change comments, add comments, or fix the code in the master branch if there are bugs. If you'd like to improve the model with techniques not used in the video, create a seperate branch for that.

### Debugging

The built in `.ToString()` method is not good for tensors, so if your trying to print a tensor while debugging, I recommend using extension method `.ToFormattedString()` that I wrote.

### NanoGpt
I was able to achieve similar results to Kaparthy. I had to use my CPU which took half a day to run. I got down to a training loss of around 0.67 and a validation loss of around 1.5. I'm not sure why my train loss and validation loss had a large spread than Kaparthy. Here's an example output of that model.

```
ICHARD III:
But look on my father's fault, but not most good.

QUEEN ELIZABETH:
Nor cheer will I thee speak; but thou hast a word, till now.

KING RICHARD III:
Then, good kind her heart; the devil
But calls me sometimes of blood for yourselves.
What service must I do? Or else thou art moved at
Her princess, having entleman of fair princely
Which I have power to dispose offender:
If I cannot discharge him, but action
That now respected, whatsoever more, title,
Constants to the world's name; and if not they see
The truth, crying but his within the fairest cover:
If the wise burning fools that they lose
The feedering steel gaoler them in sex the clouds,
You must have pair'd for sufferance.

COMINIUS:
No remedy;
That they have dropp'd from war upon you.

MENENIUS:
The gods keep you on!
There's no more!

CORIOLANUS:
No, ay, but to good with them all. You, the first
No grave for my poor general than you,--
Whether for the poor good star!

VOLUMNIA:
Not that you should not leave about the ship,--
Which you save every word of your eyes that,--
Being suffer'd with the sweets, I do notthing
Make the oseemable of strength, your enemy
Is that ever goad to be so,--think my meaning,
Nothing by that himself so shall not share
Above again him, which owe have little,
Not raged to know the cheek to the purpose,
Not summer showing: post it, sir, increaseth,
That cames to practise one would move you do,
But by that you need not, my lord.

MARCIUS:
Be not according:
The young Rome are sent forth, more strength on you,
But queen overta'en.

Messenger:
My lord,
I knees like  a Capitol
```

### Bigram Model

I was able to achieve the same, maybe slightly better results with the BigramModel. A loss of around 2.365 compared to Kaparthys 2.48.

```
step 0: train loss 4.5227, val loss 4.5216
step 300: train loss 2.7147, val loss 2.7128
step 600: train loss 2.4478, val loss 2.4512
step 900: train loss 2.4031, val loss 2.4163
step 1200: train loss 2.3846, val loss 2.3902
step 1500: train loss 2.3717, val loss 2.3847
step 1800: train loss 2.3749, val loss 2.3902
step 2100: train loss 2.3649, val loss 2.3808
step 2400: train loss 2.3697, val loss 2.3756
step 2700: train loss 2.3642, val loss 2.3853

FRe s
I mu ofr sthenck f hethefos.
Whtre IO fofel t abar alin A pr llf two s havere r makimy t o ndem'dineer.

Thisite;
HOw st hoffelan moner yonourfis aind, cinlelle IUCour gotefoulerd kl;
MI
I:
Youreitl e.
TE:
Coagarutenow.
Werd RAnes eeco nd gowhiderdw t, ty gin nde iot.
A: gre r, tak led.
IOMIswofte.
ste; wixputinte
Seile t thrand.

's ack'd pprs f thechuce tes Ye an t'ers neawas the l I wnd treme;
As'de asinge f seenthoured arisout cer
Har s, ro h, t mend Mexy aran spo
```
