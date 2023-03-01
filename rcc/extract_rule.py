# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# March 2023
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import os
from RCCSeg import RCCSeg
from rcc_common import LoadImage, SaveImage

def main():
    dataRoot="/data/AIR/RCC/NiftiNew"
    trainList=os.path.join(dataRoot, "trainList.txt")
    valList=os.path.join(dataRoot, "validationListPatient.txt")
    testList=os.path.join(dataRoot, "testListPatient.txt")

    cad = RCCSeg()

    #cad.SetDevice("cuda:0")

    cad.LoadModel("/data/AIR/RCC/ISBI/snapshots_unweighted_easyhard_deterministic_hingeforest_depth7_vggblock3_randomSplit1/epoch_549.pt")

    # Kidney
    #tree = 22
    #leaf = 15

    # Cyst
    #tree = 95
    #leaf = 67

    # Tumor
    #tree = 3
    #leaf = 27

    # From training analysis
    # Kidney
    #tree = 63
    #leaf = 4

    # Cyst
    #tree = 95
    #leaf = 67

    # Tumor
    tree = 3
    leaf = 27

    #leaf += (2**cad.net.forest.depth)-1

    path = [ ]
    rule = [ ]

    node = 0
    for d in range(cad.net.forest.depth):
        direction = (leaf & (1 << d))

        threshold = cad.net.forest.thresholds[tree,node]
        ordinal = int(cad.net.forest.ordinals[tree,node])

        path.append(node)

        if direction == 0:
            node = 2*node + 1
            rule.append("I(\\mathbf{x}_{" + f"{ordinal}" + "}" + f" \\leq {threshold:.2f})")
        else:
            node = 2*node + 2
            rule.append("I(\\mathbf{x}_{" + f"{ordinal}" + "}" + f" > {threshold:.2f})")
        
        
    """
    print(path)

    exit(0)

    path = [ ]
    rule = [ ]
    while leaf > 0:
        parent = int((leaf-1)/2)
        path = [parent] + path

        threshold = cad.net.forest.thresholds[tree,parent]
        ordinal = int(cad.net.forest.ordinals[tree,parent])

        if 2*parent + 1 == leaf:
            #rule = [ f"x_{ordinal} <= {threshold:.2f}" ] + rule
            rule = [ "I(\\mathbf{x}_{" + f"{ordinal}" + "}" + f" \\leq {threshold:.2f})" ] + rule
        elif 2*parent + 2 == leaf:
            #rule = [ f"x_{ordinal} > {threshold:.2f}" ] + rule
            rule = [ "I(\\mathbf{x}_{" + f"{ordinal}" + "}" + f" > {threshold:.2f})" ] + rule
        else:
            print("wtf?")
            exit(1)

        leaf = parent

    print(path)
    print(rule)
    """

    ruleStr = rule[0]
    for i in range(1,len(rule)):
        #ruleStr += " AND "
        ruleStr += " \\wedge "
        ruleStr += rule[i]

    print(ruleStr)

    print("Done")

if __name__ == "__main__":
    main()
