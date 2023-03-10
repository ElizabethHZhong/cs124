### Find optimal places to put line breaks in a passage based on a "preferred" number of characters/line, M. ###

WORDS = "Determine the minimal penalty, and corresponding optimal division of words into lines, for the text of this question, from the first `Determine' through the last `correctly.)', for the cases where M is 40 and M is 72. Words are divided only by spaces (and, in the pdf version, linebreaks) and each character that isn't a space (or a linebreak) counts as part of the length of a word, so the last word of the question has length eleven, which counts the period and the right parenthesis. (You can find the answer by whatever method you like, but we recommend coding your algorithm from the previous part. You don't need to submit code.) (The text of the problem may be easier to copy-paste from the tex source than from the pdf. If you copy-paste from the pdf, check that all the characters show up correctly.)"
WORDS = WORDS.split(" ")
# number of characters/line
M = 72

# M=40 -> minPenalty = 396, M=72-> minPenalty = 99

# takes in a set of words and returns the associated penalty assuming they are all on one line
def getPenalty(words, isLastLine):
    if(words == []):
        return 0
    # get the sum of characters in the words
    sumOfCharacters = sum(len(word) for word in words)

    A = M - (len(words) + sumOfCharacters - 1)
    if(A >= 0):
    #if we're at the last line of the paragraph, then no penalty if A >= 0
        if(isLastLine):
            return 0
        return A**3
    else: return 2**(-A) - A**3 - 1

breakpoints = {}
# takes in a set of words and returns the minimum penalty associated with brekaing that set of words into lines
def minPenalty(words):
    # create a dp table to store results of subproblems
    # value of dp[i] will be the minimum penalty assuming the entire sentence is just words[0:i]
    # (note that 0:0 is none of the elements and 0:1 is just the first element)
    dp = [1000000 for i in range(len(words) + 1)]

    # dp[0] is 0 because an empty string can always be made into a paragraph of 0 words.
    dp[0] = 0

    # Let's have our "lookahead" scout, i, which is always above j
    for i in range(len(words) + 1):
        # and we obvoiusly have j zooming between 0 and i
        for j in range(i):
            isLastLine = False
            if(i == len(words)):
                isLastLine = True
            penaltyBreakJ = getPenalty(words[j:i], isLastLine)
            #print("Let's say that we have a break at j (meaning we have a break between words[j-1] and words[j]): ", j)
            #print("what is the associated penalty of just the line that consists of words[j] up to words[i - 1], assuming we stop at i:", i)
            #print("getPenalty(",words[j:i],"):",penaltyBreakJ)
            #print("dp:", dp)
            #print("--------")
            # if the penalty associated with having a break at j (and obviously, stopping at i),
            # is less than the value currently at dp[i], then we should replace dp[i] with dp[j] + getPenalty(dp[i:j])
            if dp[i] > (dp[j] + penaltyBreakJ):
                dp[i] = dp[j] + penaltyBreakJ
                breakpoints[i] = j

    return dp[len(words)]

# breakpoints[len(WORDS)] is definitely a place where we should have a breakpoint. This is because it's the place where having a breakpoint minimizes the penalty assuming that our entire sentence is WORDS (which it is).
# So if it's 4, then we have a break bewtween words[3] and words[4]
# Therefore, we know we should have a breakpoint at breakpoints[len(WORDS)]
# Given that we know we should have a brekapoint at breakpoints[len(WORDS)], then we know that the optimal next breakpoint
# is at breakpoints[breakpoints[len(WORDS)]]. And so on and so forth until we get to the first breakpoint.
print("minimum penalty for your sentence: ", minPenalty(WORDS))

new_idx = breakpoints[len(WORDS)]
line_breaks = [new_idx]
while breakpoints[new_idx] != 0:
    new_idx = breakpoints[new_idx]
    line_breaks.append(new_idx)

# reverse line_breaks
line_breaks.reverse()
print("line breaks at the following locations: ", line_breaks)

# print the paragraph with the line breaks
for i in range(len(line_breaks)):
    if(i == 0):
        print(" ".join(WORDS[0:line_breaks[i]]))
    else:
        print(" ".join(WORDS[line_breaks[i-1]:line_breaks[i]]))
print(" ".join(WORDS[line_breaks[-1]:len(WORDS)]))