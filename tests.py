"""This module contains tests of the zernike module. 

Author: Ravi S. Jonnal / Werner Lab, UC Davis

Revision: 2.0 / 28 June 2014

"""

from zernike import Zernike
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from time import sleep
import os

def writeTex(filename='./polynomials/polynomials.tex',maxOrder=10):

    head,tail = os.path.split(filename)
    if not os.path.exists(head):
        print 'Making directory %s.'%head
        os.makedirs(head)
    
    z = Zernike()
    fh = open(filename,'wb')
    head = "\documentclass[10pt,landscape]{article}\n\usepackage[margin=0.25in]{geometry}\n\\begin{document}\n\n"
    intro = "\\noindent The equations below are generated using \\texttt{Zernike.getEquationString} method.\n\n"

    fh.write(head)
    fh.write(intro)
    for tup in [('h','Height ($Z$)'),('dx','X slope ($\\frac{\delta Z}{\delta X}$)'),('dy','Y slope ($\\frac{\delta Z}{\delta Y}$)')]:

        fh.write("\section{%s}\n\n"%(tup[1]))
        for n in range(maxOrder+1):
            for m in range(-n,n+1,2):
                fh.write('\\noindent %s\n\n'%(z.getEquationString(n,m,tup[0])))
                fh.write('\\vspace{1.2 mm}\n')
        fh.write("\clearpage\n\n")

    fh.write("\n\end{document}")
    fh.close()


def plotAnimation(maxOrder=5):

    """A test function for animated visualization of a subset of the
    polynomial surfaces specified by maxOrder.
    """
    if not os.path.exists('images'):
        print 'Making directory images.'
        os.makedirs('images')
    
    z = Zernike()
    print 'Plotting first %d orders of Zernike polynomial surfaces.'%(maxOrder+1)
    order = maxOrder+1

    fig = plt.figure()
    plt.ion()

    for j in range(order):
        for k in range(-j,j+1,2):
            z.plotPolynomial(j,k)
            plt.savefig('./images/mode_%d_%d.png'%(j,k))
            plt.draw()
            sleep(.1)
            plt.cla()
            plt.clf()

    plt.ioff()
    plt.close()

def simulateShackHartmannReconstruction():
    """A test function that uses Zernike modes to generate a random simulated wavefront, and then
    uses Zernike derivatives to estimate the original coefficients.
    """
    pupilDiameterM = 10e-3

    pupilRadiusM = pupilDiameterM / 2.0
    unity = 1.0

    z = Zernike()

    nZT = 1+2+3+4+5+6+7

    # Set up some aligned vectors of Zernike index (j), order (n),
    # frequency (m), and tuples of order and frequency (nm).
    jVec = range(nZT)
    nmVec = np.array([z.j2nm(x) for x in jVec])
    nVec = np.array([z.j2nm(x)[0] for x in jVec])
    mVec = np.array([z.j2nm(x)[1] for x in jVec])
    maxOrder = nVec.max()

    # Generate some random Zernike coefficients. Seed the generator to
    # permit iterative comparison of output. Zero the first three modes
    # (piston, tip, tilt).
    
    cVec = np.random.randn(len(jVec))/10000.0
    cVec[:3] = 0.0 # zero tip, tilt, and piston

    # We'll make a densely sampled wavefront using the modes and
    # coefficients specified above.
    N = 1000
    xx,yy = np.meshgrid(np.linspace(-unity,unity,N),np.linspace(-unity,unity,N))
    mask = np.zeros((N,N)) # circular mask to define unit pupil
    d = np.sqrt(xx**2 + yy**2)
    mask[np.where(d<unity)] = 1
    wavefront = np.zeros(mask.shape) # matrix to accumulate wavefront error

    # We build up the simulated wavefront by summing the modes of
    # interest. The simulated wavefront has units of pupil radius; to
    # convert to meters, we have to multiply by the pupil radius in
    # meters.
    for (n,m),coef in zip(nmVec,cVec):
        print 'Adding %0.6f Z(%d,%d) to simulated wavefront.'%(coef,n,m)
        wavefront = wavefront + coef*z.getSurface(n,m,xx,yy,'h')*pupilRadiusM

    def coord2index(coord):
        test = xx[0,:]
        return np.argmin(np.abs(test-coord))


    # In order to compute the slope at each subaperture, we need to know
    # the pixel size.
    pixelSizeM = pupilDiameterM/N

    # Now we'll define a lenslet array as a masked 20x20 grid. Since we
    # want the edges of the outer lenslets to abutt the pupil edge, let's
    # start by enumerating the 21 edge coordinates in the unit line:
    nLensletsAcross = 20
    lensletEdges = np.linspace(-unity,unity,nLensletsAcross+1)
    # The centers can now be specified as the average between the leftmost
    # 20 coordinates and the rightmost 20 coordinates:
    lensletCenters = (lensletEdges[1:]+lensletEdges[:-1])/2.0
    lensletXX,lensletYY = np.meshgrid(lensletCenters,lensletCenters)
    lensletD = np.sqrt(lensletXX**2+lensletYY**2)
    lensletMask = np.zeros(lensletD.shape)
    lensletMask[np.where(lensletD<unity)]=1

    # Pull out the x and y coordinates of the lenslets into vectors, and
    # we'll iterate through these to build corresponding vectors of local
    # slopes:
    lensletXXVector = lensletXX[np.where(lensletMask)]
    lensletYYVector = lensletYY[np.where(lensletMask)]

    # Estimate the local slope in each subaperture:
    xSlopes = []
    ySlopes = []
    for x,y in zip(lensletXXVector,lensletYYVector):
        print 'Estimating slope at (%0.2f,%0.2f).'%(x,y)
        x1i = coord2index(x-.5/nLensletsAcross)
        x2i = coord2index(x+.5/nLensletsAcross)
        y1i = coord2index(y-.5/nLensletsAcross)
        y2i = coord2index(y+.5/nLensletsAcross)
        subap = wavefront[y1i:y2i,x1i:x2i]
        dSubapX_dPx = np.diff(subap,axis=1).mean()/pixelSizeM
        dSubapY_dPx = np.diff(subap,axis=0).mean()/pixelSizeM
        xSlopes.append(dSubapX_dPx)
        ySlopes.append(dSubapY_dPx)

    xSlopes = np.array(xSlopes)
    ySlopes = np.array(ySlopes)
    nLenslets = len(xSlopes)


    # Now that we have a simulated wavefront and simulated slope
    # measurements, let's reconstruct!  Reconstruction consists of:
    #
    #   1) Using zernike.Zernike.getSurface to calculate the x
    #      and y derivatives of each Zernike mode; 
    #
    #   2) Using np.where and lensletMask to order the unmasked
    #      derivatives the same way they're ordered in lensletXXVector and
    #      lensletYYVector above;
    #
    #   3) Assembling these derivatives into a matrix A such that A[n,m]
    #      contains the partial x-derivative of the mth mode at the
    #      location corresponding to the nth lenslet. A[n+k,m] is the
    #      corresponding partial y-derivative, where k is the number of
    #      lenslets.
    #
    #   4) Inverting A and multiplying it by the local slopes gives us
    #      reconstructed Zernike coefficients, which can then be compared
    #      with our randomly chosen initial values.

    A = np.zeros([nLenslets*2,nZT])

    for j in jVec:
        iOrder,iFreq = z.j2nm(j)
        print 'Calculating Zernike derivatives for Z(%d,%d).'%(iOrder,iFreq)
        dzdx = z.getSurface(iOrder,iFreq,lensletXX,lensletYY,'dx')
        dzdy = z.getSurface(iOrder,iFreq,lensletXX,lensletYY,'dy')
        dzdx = dzdx[np.where(lensletMask)]
        dzdy = dzdy[np.where(lensletMask)]
        A[:nLenslets,j] = dzdx
        A[nLenslets:,j] = dzdy

    slopes = np.hstack((xSlopes,ySlopes))

    # Sidebar:
    # Note that once the matrix A is built, we can trivially compute the
    # slopes at our desired locations by simply computing the dot product
    # of A and our Zernike coefficient vector cVec.
    # Let's do this for fun:
    dotSlopes = np.dot(A,cVec)
    plt.figure(figsize=(12,6))
    plt.plot(slopes,dotSlopes,'ks')
    plt.xlabel('slope estimated by local curvature')
    plt.ylabel('slope calculated by Zernike derivatives')
    plt.savefig('./images/slope_comparison.png')


    # Back to the reconstruction. Now we use SVD to invert A and
    # multiply the resulting inverse by the slopes to determine the
    # modal coefficients:
    B = np.linalg.pinv(A)
    rVec = np.dot(B,slopes)


    # Using the reconstructed coefficients, we can reconstruct the
    # wavefront.
    rWavefront = np.zeros(wavefront.shape)
    for (n,m),coef in zip(nmVec,rVec):
        rWavefront = rWavefront + coef*z.getSurface(n,m,xx,yy,'h')*pupilRadiusM


    # Now we plot the results.
    fig = plt.figure(figsize=(12,12))

    fig.add_subplot(2,2,1)
    plt.imshow(wavefront*mask)
    for x,y in zip(lensletXXVector,lensletYYVector):
        plt.plot(coord2index(x),coord2index(y),'ks')

    plt.axis('image')
    plt.axis('tight')
    plt.title('simulated wavefront and sampling locations')

    fig.add_subplot(2,2,2)
    comparison_plot_type='bar'
    xax = np.arange(len(cVec))
    if comparison_plot_type=='scatter':
        ph1=plt.plot(xax,cVec,'gs')[0]
        ph1.set_markersize(8)
        ph1.set_markeredgecolor('k')
        ph1.set_markerfacecolor('g')
        ph1.set_markeredgewidth(2)

        ph2=plt.plot(xax+.35,rVec,'bs')[0]
        ph2.set_markersize(8)
        ph2.set_markeredgecolor('k')
        ph2.set_markerfacecolor('g')
        ph2.set_markeredgewidth(2)

        err = np.sqrt(np.mean((cVec[3:]-rVec[3:])**2))
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        plt.text(xlim[1],ylim[0],'$\epsilon=%0.2e$'%err,ha='right',va='bottom')

        plt.legend(['simulated','reconstructed'])

    elif comparison_plot_type=='bar':
        bar_width = .45
        rects1 = plt.bar(xax,cVec,bar_width,color='g',label='simulated')
        rects2 = plt.bar(xax + bar_width,rVec,bar_width,color='b',label='reconstructed')
        plt.legend()
        plt.tight_layout()

    plt.title('reconstructed zernike terms')
    plt.ylabel('coefficient')

    xticks = plt.gca().get_xticks()
    xticklabels = []
    for xtick in xticks:
        try:
            n = nVec[xtick]
            m = mVec[xtick]
            xticklabels.append('$Z^{%d}_{%d}$'%(m,n))
        except Exception as e:
            pass

    plt.gca().set_xticklabels(xticklabels)

    ax = fig.add_subplot(2,2,3,projection='3d')
    surf = ax.plot_wireframe(1e3*xx*pupilRadiusM,1e3*yy*pupilRadiusM,1e6*wavefront*mask,rstride=50,cstride=50,color='k')
    ax.view_init(elev=43., azim=68)
    plt.title('simulated wavefront')
    plt.xlabel('pupil x (mm)')
    plt.ylabel('pupil y (mm)')
    ax.set_zlabel('height ($\mu m$)')

    ax = fig.add_subplot(2,2,4,projection='3d')
    surf = ax.plot_wireframe(1e3*xx*pupilRadiusM,1e3*yy*pupilRadiusM,1e6*rWavefront*mask,rstride=50,cstride=50,color='k')
    ax.view_init(elev=43., azim=68)
    plt.title('reconstructed wavefront')
    plt.xlabel('pupil x (mm)')
    plt.ylabel('pupil y (mm)')
    ax.set_zlabel('height ($\mu m$)')

    plt.savefig('./images/shack_hartmann_reconstruction_simulation.png')



if __name__=='__main__':
    
    # Plot the first 10 Zernike modes (orders 0, 1, 2, 3).
    plotAnimation(3)

    # Write some Zernike mode terms to tex file.
    polynomialTexFile = './polynomials/polynomials.tex'
    maxOrder = 15
    print 'Writing equations up to order %d to %s.'%(maxOrder,polynomialTexFile)
    writeTex(polynomialTexFile,maxOrder)

    # Simulate Shack Hartmann sampling of a wavefront slope and subsequent
    # Zernike reconstruction. See tests.py for details.
    simulateShackHartmannReconstruction()


    plt.show()


