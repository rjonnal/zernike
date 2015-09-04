"""This module contains functions for Zernike calculations. Mainly the private
function _zgen, a generator function for Zernike polynomials. The public
functions make use of _zgen to create height or slope maps in a unit
pupil, corresponding to individual Zernike terms.

Author: Ravi S. Jonnal / Werner Lab, UC Davis

Revision: 2.0 / 28 June 2014

"""

import numpy as np
from matplotlib import pyplot as plt
import sys
from time import sleep
import os

USE_CACHE_FILE = False

def fact(num):
    """Implementation of factorial function.
    """
    # Check that the number is an integer.
    assert(num%1==0)
    # Check that $num\geq 0$.
    assert(num>=0)
    
    # Compute $num!$ recursively.
    if num==0 or num==1:
        return 1
    else:
        return num * fact(num-1)


def choose(a,b):
    """Binomial coefficient, implemented using
    this module's factorial function.
    See [here](http://www.encyclopediaofmath.org/index.php/Newton_binomial) for detail.
    """
    assert(a>=b)
    return fact(a)/(fact(b)*fact(a-b))

def splitEquation(eqStr,width,bookend):
    if len(eqStr)<=width or len(eqStr)==0:
        return eqStr
    else:
        spaceIndices = []
        idx = 0
        while idx>-1:
            idx = eqStr.find(' ',idx+1)
            spaceIndices.append(idx)
        spaceIndices = spaceIndices[:-1]
        idxList = [x for x in spaceIndices if x<width]
        if len(idxList)==0:
            return eqStr
        else:
            idx = idxList[-1]
            head = eqStr[:idx]
            innards = ' ' + bookend + '\n' + bookend
            tail = splitEquation(eqStr[idx:],width,bookend)
            test =head + innards + tail
            return test
    
class Zernike:

    def __init__(self):
        
        if USE_CACHE_FILE:
            cachedir = './cache/'
            self._cachefn = os.path.join(cachedir,'zernike_cache.txt')
            
            if not os.path.exists(cachedir):
                os.makedirs(cachedir)

            try:
                self._termMatrix = np.loadtxt(self._cachefn).astype(np.int32)
            except Exception as e:
                print 'No term cache file. Creating.'
                self._termMatrix = np.array([])
                np.savetxt(self._cachefn,self._termMatrix)


        # Make a dictionary of precomputed coefficients, using the cache file.
        # This dictionary will be used to look up values when they exist in
        # the dictionary, and will recompute them otherwise.
        self._termDict = {}

        if USE_CACHE_FILE:
            for row in self._termMatrix:
                n,m,kindIndex,s,j,k = row[:6]
                t1,t2,t3,c,tXexp,tYexp = row[6:]
                self._termDict[(n,m,kindIndex,s,j,k)] = (t1,t2,t3,c,tXexp,tYexp)


        # The functions in this class can be asked for phase height,
        # or partial x or partial y derivatives. 'Kind' refers to
        # which of these is requested. Numerical encodings for 'kind'
        # permit some arithmetical simplicity and generality
        # (utilizing a number associated with the kind in a single
        # equation, rather than having different sets of equations
        # for each kind case).
        self._kindDictionary = {}
        self._kindDictionary['h'] = 0
        self._kindDictionary['dx'] = 1
        self._kindDictionary['dy'] = 2


    def j2nm(self,j):
        n = np.ceil((-3+np.sqrt(9+8*j))/2)
        m = 2*j-n*(n+2)
        return np.int(n),np.int(m)

    def nm2j(self,n,m):
        return np.int(n*(n+1)/2.0+(n+m)/2.0)


    def _zeqn(self,n,m,kind='h',forceRecompute=False):
        """Return parameters sufficient for specifying a Zernike term
        of desired order and azimuthal frequency.

        Given an order (or degree) n and azimuthal frequency f, and x-
        and y- rectangular (Cartesian) coordinates, produce parameters
        necessary for constructing the appropriate Zernike
        representation.

        An individual polynomial has the format:

        $$ Z_n^m = \sqrt{c} \Sigma^j\Sigma^k [a_{jk}X^jY^k] $$

        This function returns a tuple ($c$,cdict). $c$ is the square
        of the normalizing coefficient $\sqrt{c}$, and cdict contains
        key-value pairs (($j$,$k$),$a$), mapping the $X$ and $Y$
        exponents ($j$ and $k$, respectively) onto polynomial term
        coefficients ($a$). The resulting structure can be used to
        compute the wavefront height or slope for arbitrary pupil
        coordinates, or to generate string representations of the
        polynomials.

        Zernike terms are only defined when n and m have the same
        parity (both odd or both even).

        Please see Schwiegerling lecture notes in
        /doc/supporting_docs/ for eqn. references.

        Args:

          n (int): The Zernike order or degree.

          m (int): The azimuthal frequency.

          kind (str): 'h', 'dx', or 'dy', for height, partial x
              derivative (slope) or partial y derivative,
              respectively.

        Returns:

          params (tuple): (c,cdict), with c being the normalizing
              coefficient c and cdict being the map of exponent pairs
              onto inner coefficients.

        """
        absm = np.abs(m)

        kindIndex = self._kindDictionary[kind.lower()]

        if USE_CACHE_FILE:
            # open cache file in append mode:
            self._cacheHandle = file(self._cachefn,'a')
        
        # check that n and m are both even or both odd
        if (float(n-absm))%2.0:
            errString = 'zernike._zgen error: ' + \
                'parity of n and m are different; n = %d, m = %d'%(n,m)
            sys.exit(errString)

        # check that n is non-negative:
        if n<0:
            errString = 'zernike._zgen error: ' + \
                'n must be non-negative; n = %d'%n
            sys.exit(errString)

        # $|m|$ must be less than or equal to $n$.
        if abs(m)>n:
            errString = 'zernike._zgen error: ' + \
                '|m| must be less than or equal to n, but n=%d and m=%d.'%(n,m)
            sys.exit(errString)

        # These are the squares of the outer coefficients. It's useful
        # to keep them this way for _convertToString, since we'd
        # prefer to print the $\sqrt{}$ rather than a truncated irrational
        # number.
        if m==0:
            outerCoef = n+1
        else:
            outerCoef = 2*(n+1)


        srange = range((n-absm)/2+1)

        cdict = {}

        for s in srange:
            jrange = range(((n-absm)/2)-s+1)
            for j in jrange:

                # Subtract 1 from absm to determine range,
                # only when m<0.
                if m<0:
                    krange = range((absm-1)/2+1)
                else:
                    krange = range(absm/2+1)

                for k in krange:
                    # If m==0, k must also be 0;
                    # see eqn. 13c, 19c, and 20c, each of which
                    # only sum over s and j, not k.
                    if m==0:
                        assert(k==0)
                    # For m==0 cases, n/2 is used in coef denominator. Make
                    # sure that n is even, or else n/2 is not well-defined
                    # because n is an integer.
                    if m==0:
                        assert n%2==0
                    
                    
                    # Check to see if calculations are cached.
                    # If so, use cached values; if not, recalculate.
                    cached = self._termDict.has_key((n,m,kindIndex,s,j,k))

                    if cached and not forceRecompute:
                        t1,t2,t3,c,tXexp,tYexp = self._termDict[(n,m,kindIndex,s,j,k)]
                        
                    else:

                        # The coefficient for each term in this
                        # polynomial has the format: $$\frac{t1n}{t1d1
                        # t1d2 t1d3} t2 t3$$. These six terms are
                        # computed here.
                        t1n = ((-1)**(s+k))*fact(n-s)
                        t1d1 = fact(s)
                        t1d2 = fact((n + absm)/2-s)
                        t1d3 = fact((n - absm)/2-s)
                        t1 = t1n/(t1d1*t1d2*t1d3)
                        
                        t2 = choose((n - absm)/2 - s, j)
                        t3 = choose(absm, 2*k + (m<0))


                        if kind.lower()=='h':
                            # The (implied) coefficient of the $X^a Y^b$
                            # term at the end of eqns. 13a-c.
                            c = 1 
                            tXexp = n - 2*(s+j+k) - (m<0)
                            tYexp = 2*(j+k) + (m<0)
                        elif kind.lower()=='dx':
                            # The coefficient of the $X^a Y^b$ term at
                            # the end of eqns. 19a-c.
                            c = (n - 2*(s+j+k) - (m<0)) 

                            # Could cacluate explicitly:
                            # $tXexp = X^{(n - 2*(s+j+k)- 1 - (m<0))}$
                            # 
                            # However, piggy-backing on previous
                            # calculation of c speeds things up.
                            tXexp = c - 1
                            tYexp = 2*(j+k) + (m<0)

                        elif kind.lower()=='dy':
                            # The coefficient of the $X^a Y^b$ term at
                            # the end of eqns. 20a-c.
                            c = 2*(j+k) + (m<0)
                            tXexp = n - 2*(s+j+k) - (m<0)
                            tYexp = c - 1

                        else:
                            errString = 'zernike._zgen error: ' + \
                                'invalid kind \'%s\'; should be \'h\', \'dx\', or \'dy\'.'%kind
                            sys.exit(errString)

                        if not cached and USE_CACHE_FILE:
                            self._cacheHandle.write('%d\t'*12%(n,m,kindIndex,s,j,k,t1,t2,t3,c,tXexp,tYexp)+'\n')

                    ct123 = c*t1*t2*t3
                    # The key for the polynomial dictionary is the pair of X,Y
                    # coefficients.
                    termKey = (tXexp,tYexp)

                    # Leave this term out of the dictionary if its coefficient
                    # is 0.
                    if ct123:
                        # If we already have this term, add to its coefficient.
                        if cdict.has_key(termKey):
                            cdict[termKey] = cdict[termKey] + ct123
                        # If not, add it to the dictionary.
                        else:
                            cdict[termKey] = ct123

        # Remove zeros to speed up computations later.
        cdict = {key: value for key, value in cdict.items() if value}

        return (outerCoef,cdict)

    def _convertToString(self,params):
        """Return a string representation of a Zernike polynomial.

        This function takes a tuple, consisting of a squared
        normalizing coefficient and dictionary of inner coefficients
        and exponents, provided by _zeqn, and returns a string
        representation of the polynomial, with LaTeX- style markup.

        Example: a params of (10, {(3,4): 7, (2,5): -1}) would produce a
        two-term polynomial '\sqrt{10} [7 X^3 Y^4 - X^2 Y^5]', which could be used in LaTeX,
        pandoc, markdown, MathJax, or Word with MathType, to produce:

        $$ \sqrt{10} [7 X^3 Y^4 - X^2 Y^5] $$

        Args:

          params (tuple): A pair consisting of an outer coefficient
            $c$ and a dictionary mapping tuples (xexp,yexp) of
            exponents onto the corresponding term coefficients.
            

        Returns:

          string: A string representation of the polynomial.
        """
        c = params[0]
        cdict = params[1]


        keys = sorted(cdict.keys(), key=lambda tup: (tup[0]+tup[1],tup[0]))[::-1]

        outstr = ''
        firstKey = True
        for key in keys:
            coef = cdict[key]
            if coef>0:
                sign = '+'
            else:
                sign = '-'

            coef = abs(coef)
            
            if coef<0 or not firstKey:
                outstr = outstr + '%s'%sign
            if coef>1 or (key[0]==0 and key[1]==0):
                outstr = outstr + '%d'%coef
            if key[0]:
                outstr = outstr + 'X^{%d}'%key[0]
            if key[1]:
                outstr = outstr + 'Y^{%d}'%key[1]
            firstKey = False
            outstr = outstr + ' '

        outstr = outstr.strip()
        

        if np.sqrt(float(c))%1.0<.00001:
            cstr = '%d'%(np.sqrt(c))
        else:
            cstr = '\sqrt{%d}'%(c)

        if len(outstr):
            outstr = '%s [%s]'%(cstr,outstr)
        else:
            outstr = '%s'%(cstr)
        return outstr

    def _convertToSurface(self,params,X,Y,mask=None):
        """Return a phase map specified by a Zernike polynomial.

        This function takes a tuple, consisting of a squared
        normalizing coefficient and dictionary of inner coefficients
        and exponents, provided by _zeqn, and x- and y- rectangular
        (Cartesian) coordinates, and produces a phase map.

        This function works by evaluating the polynomial expressed by
        params at each coordinate specified by X and Y.


        Args:

          params (tuple): A pair consisting of an outer coefficient
            $c$ and a dictionary mapping tuples (xexp,yexp) of
            exponents onto the corresponding term coefficients.

          X (float): A scalar, vector, or matrix of X coordinates in unit pupil.

          Y (float): A scalar, vector, or matrix of Y coordinates in unit pupil.

          kind (str): 'h', 'dx', or 'dy', for height, partial x derivative (slope)
              or partial y derivative, respectively.

        Returns:

          float: height, dx, or dy; returned structure same size as X and Y.
        """

        # Check that shapes of X and Y are equal (not necessarily square).
        if not (X.shape[0]==Y.shape[0] and \
                    X.shape[1]==Y.shape[1]):
            errString = 'zernike.getSurface error: ' + \
                'X and Y must have the same shape, but X is %d x %d'%(X.shape[0],X.shape[1]) + \
                'and Y is %d x %d'%(Y.shape[0],Y.shape[1])
            sys.exit(errString)

        if mask is None:
            mask = np.ones(X.shape)

        params = self._zeqn(n,m,kind)
        normalizer = np.sqrt(params[0])

        matrix_out = np.zeros(X.shape)
        

        for item in params[1].items():
            matrix_out = matrix_out + item[1] * X**(item[0][0]) * Y**(item[0][1])

        matrix_out = matrix_out * np.sqrt(normalizer)
        matrix_out = matrix_out * mask

        return matrix_out

    def getSurface(self,n,m,X,Y,kind='h',mask=None):
        """Return a phase map specified by a Zernike order and azimuthal frequency.

        Given an order (or degree) n and azimuthal frequency f, and x- and y-
        rectangular (Cartesian) coordinates, produce a phase map of either height,
        partial x derivative, or partial y derivative.

        Zernike terms are only defined when n and m have the same parity (both odd
        or both even).

        The input X and Y values should be located inside a unit pupil, such that
        $$\sqrt{X^2 + Y^2}\leq 1$$

        Please see Schwiegerling lecture notes in /doc/supporting_docs/ for eqn.
        references.

        This function works by calling Zernike._zeqn to calculate the coefficients
        and exponents of the polynomial, and then using the supplied X and Y
        coordinates to produce the height map (or partial derivative).

        Args:

          n (int): The Zernike order or degree.

          m (int): The azimuthal frequency.

          X (float): A scalar, vector, or matrix of X coordinates in unit pupil.

          Y (float): A scalar, vector, or matrix of Y coordinates in unit pupil.

          kind (str): 'h', 'dx', or 'dy', for height, partial x derivative (slope)
              or partial y derivative, respectively.

        Returns:

          float: height, dx, or dy; returned structure same size as X and Y.
        """

        # Check that shapes of X and Y are equal (not necessarily square).
        if not np.all(X.shape==Y.shape):
            errString = 'zernike.getSurface error: ' + \
                'X and Y must have the same shape, but X is %d x %d'%(X.shape[0],X.shape[1]) + \
                'and Y is %d x %d'%(Y.shape[0],Y.shape[1])
            sys.exit(errString)
        

        if mask is None:
            mask = np.ones(X.shape)

        params = self._zeqn(n,m,kind)
        normalizer = np.sqrt(params[0])
        matrix_out = np.zeros(X.shape)
        

        for item in params[1].items():
            matrix_out = matrix_out + item[1] * X**(item[0][0]) * Y**(item[0][1])

        matrix_out = matrix_out * normalizer
        matrix_out = matrix_out * mask

        return matrix_out


    def getEquationString(self,n,m,kind='h',doubleDollar=False):
        """Return LaTeX-encoded of the Zernike polynomial specified by
        order n, frequency m.

        Args:
        
          n (int): The Zernike order or degree.

          m (int): The azimuthal frequency.
          
          kind (str): 'h', 'dx', or 'dy', for height, partial x
              derivative (slope) or partial y derivative,
              respectively.

          doubleDollar (bool): determines how to bookend the
              polynomial string; True causes bookending with '$$', to
              produce "display" math mode, whereas False would produce
              a string suitable for inline use.

        Returns:

          str: a LaTeX representation of the Zernike polynomial
              specified by n, m, and Kind.
        """

        params = self._zeqn(n,m,kind)
        rightString = self._convertToString(params)

        if kind.lower()=='h':
            leftString = 'Z^{%d}_{%d}'%(m,n)
        elif kind.lower()=='dx':
            leftString = '\\frac{\delta Z^{%d}_{%d}}{\delta x}'%(m,n)
        elif kind.lower()=='dy':
            leftString = '\\frac{\delta Z^{%d}_{%d}}{\delta y}'%(m,n)
        else:
            sys.exit('zernike.getEquationString: invalid kind %s'%kind)


        if doubleDollar:
            bookend = '$$'
        else:
            bookend = '$'

        return '%s %s = %s %s'%(bookend,leftString,rightString,bookend)


    def plotPolynomial(self,n,m,kind='h'):
        """Plot a polynomial surface specified by order n, frequency m, and kind.

        Args:
        
          n (int): The Zernike order or degree.

          m (int): The azimuthal frequency.
          
          kind (str): 'h', 'dx', or 'dy', for height, partial x
              derivative (slope) or partial y derivative,
              respectively.

        Calling function/script required to provide a plotting context (e.g. pyplot.figure).
        """
        
        from mpl_toolkits.mplot3d import Axes3D

        N = 64
        mask = np.zeros((N,N))
        xx,yy = np.meshgrid(np.linspace(-1,1,N),np.linspace(-1,1,N))
        d = np.sqrt(xx**2 + yy**2)
        mask[np.where(d<1)] = 1
        surface = self.getSurface(n,m,xx,yy,kind,mask)

        surface = surface * mask
        #plt.figure()
        ax = plt.axes([0,.2,1,.8],projection='3d')
        surf = ax.plot_wireframe(xx,yy,surface,rstride=1,cstride=1,color='k')
        ax.view_init(elev=70., azim=40)

        eqstr = self.getEquationString(n,m,kind)
        eqstr = splitEquation(eqstr,160,'$')
        print 'plotting %s'%eqstr
        plt.axes([0,0,1,.2])
        plt.xticks([])
        plt.yticks([])
        plt.box('off')
        fontsize = 12
        plt.text(0.5,0.5,eqstr,ha='center',va='center',fontsize=fontsize)
        


