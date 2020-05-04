""" Description
Writea subroutine“distance(x1,x2)”to compute the distance between two data instances x1,x2. 
Inside distance(), you should group numerical attributes together as num-set, group categorical attribute together as cat-set, etc. 
On num-set, you use Euclidean distance. On cat-set, you use Hamming distance.
(For example, we have x1 = [11, “warm-blooded”, 2] and x2 = [13, “cold-blooded”, 3], distance(x1,x2) will be(11-13)2+ 1 + (2-3)2= 6.)
"""
def distance(x1, x2):
    # Input can be numer-set/group categorical attribute
    dist = 0
    for i in range(len(x1)):
        if str(x1[i]).isdigit():
            dist += (x1[i]-x2[i])**2
        else:
            w1, w2 = x1[i].split('-'), x2[i].split('-')
            dist += sum([1 if w1[i]==w2[i] else 0 for i in range(len(w1))])
    return dist
    
if __name__ == "__main__":
    dist = distance([11, "warm-blooded", 2], [13, "cold-blooded", 3])
    print(dist)
        