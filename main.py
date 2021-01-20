import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


SAMPLESIZE = 2
SEQUENCESIZE = 200

# number of sequences we will be finding variance for
SEQUENCECOUNT = 20

MAXSEQUENCENUMBER = 10


def getsample(sequence):
    sample = list(sequence)
    np.random.shuffle(sample)
    return sample[:SAMPLESIZE]


def variance(sample, n, correction=True):
    """
    sample: tuple
    n: int (sample size)
    correction: bool (defaults to True, as in unbiased sample variance)
    """

    squared_sum = sum(map(lambda val: val**2, sample))

    if correction:
        return (n*squared_sum - sum(sample)**2) / (n - 1) / n
    else:
        return (n*squared_sum - sum(sample)**2) / n**2


def getinfo(sequence):
    """
    returns a tuple of population variance, unbiased sample variance, biased sample variance
    errors of each sample variance
    """

    sample = getsample(sequence)
    actual = variance(sequence, len(sequence), correction=False)
    unbiased = variance(sample, SAMPLESIZE, correction=True)
    biased = variance(sample, SAMPLESIZE, correction=False)

    # add info about whether the biased or unbiased is closer to actual
    # calculate the errors for each estimate

    unbiased_error = abs(1 - unbiased/actual)
    biased_error = abs(1 - biased/actual)

    return tuple((actual, unbiased, biased, unbiased_error, biased_error))


def sequencegen(randomrange, seqlen, type):
    """
    randomrange: tuple (2)
    seqlen: int
    generates a random tuple of length and randomrange
    """

    minval = randomrange[0]
    maxval = randomrange[1]

    if type == "random":
        return tuple(np.random.randint(minval, maxval) for i in range(seqlen))

    if type == "normal":
        mean = (minval + maxval)/2

        # 99.7% of the data will lie between min and max
        # the width of the range should be six sigma
        sigma = (maxval - minval) / 6

        return tuple(np.random.normal(loc=mean, scale=sigma) for i in range(seqlen))


def getdataframe(type):
    sequences = tuple(
        sequencegen((0, MAXSEQUENCENUMBER), SEQUENCESIZE, type) for i in range(SEQUENCECOUNT)
    )

    df = pd.DataFrame(
        map(getinfo, sequences),
        columns=("actual", "unbiased", "biased",
                 "unbiased_error", "biased_error")
    )

    return df


#### GRAPHING SETTINGS ###
fig, axes = plt.subplots(nrows=3, ncols=2)

# graph title
title = "Comparing population variance with biased and unbiased sample variance"
fig.suptitle(title)

# random sequence in left column, normal sequence in right column
axes[0][0].set_title("random sequence")
axes[0][1].set_title("random-normal sequence")

# adjusting whitepace of the plots
plt.subplots_adjust(hspace=0.47)

# adjusting figure size
fig.set_size_inches(13, 8)

# changing the error axes y-label to percents
for ax in axes[1][:2]:
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

# set label for the bar charts
plt.setp(axes[:2][:], xlabel="sequence index")

# set label for the pie charts
plt.setp(axes[2][:], xlabel="total error")

##########################

### GENERATING DATA AND PLOTTING ###

# top row: variances
df_random = getdataframe("random")
df_random_variances = df_random[["actual", "unbiased", "biased"]]
df_random_variances.plot(kind="bar", ax=axes[0][0], color=["b", "g", "r"])

df_normal = getdataframe("normal")
df_normal_variances = df_normal[["actual", "unbiased", "biased"]]
df_normal_variances.plot(kind="bar", ax=axes[0][1], color=["b", "g", "r"])

# middle row: errors of biased and unbiased sample variance
df_random_errors = df_random[["unbiased_error", "biased_error"]]
df_random_errors.plot(kind="bar", ax=axes[1][0], color=["g", "r"])

df_normal_errors = df_normal[["unbiased_error", "biased_error"]]
df_normal_errors.plot(kind="bar", ax=axes[1][1], color=["g", "r"])

# last row: total error
total_unbiased_error_random = float(df_random_errors[["unbiased_error"]].sum())
total_biased_error_random = float(df_random_errors[["biased_error"]].sum())
axes[2][0].pie([total_unbiased_error_random,
                total_biased_error_random], colors=["g", "r"])

total_unbiased_error_normal = float(df_normal_errors[["unbiased_error"]].sum())
total_biased_error_normal = float(df_normal_errors[["biased_error"]].sum())
axes[2][1].pie([total_unbiased_error_normal,
                total_biased_error_normal], colors=["g", "r"])

plt.show()

#####################################
