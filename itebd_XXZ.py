#!/usr/bin/python3
"""
Usage:
    itebd_XXZ.py waveInfo <waveFile> [ --delta DELTA -o OFILE ]
    itebd_XXZ.py contin <waveFile> [ -f PARAMETERFILE -o OFILE --noOutput ]
    itebd_XXZ.py [ -f PARAMETERFILE -o OFILE -w WAVEFILE --noOutput ]
    itebd_XXZ.py -h | --help

Options:
    -h, --help
    -f PARAMETERFILE
    -o OFILE
    -w WAVEFILE
    --delta DELTA   [default: 1.0]
"""

import scipy as sp
import scipy.linalg as slg
from time import time
from sys import stderr, stdout, exit
from docopt import docopt

sx = 0.5 * sp.array( [ [ 0.0, 2.0, 0.0, 0.0, 0.0 ],
                 [ 2.0, 0.0, 6 ** 0.5, 0.0, 0.0 ],
                 [ 0.0, 6 ** 0.5, 0.0, 6 ** 0.5, 0.0 ],
                 [ 0.0, 0.0, 6 ** 0.5, 0.0, 2.0 ],
                 [ 0.0, 0.0, 0.0, 2.0, 0.0 ] ], dtype = sp.complex128 )

sy = -0.5j * sp.array( [ [ 0.0, 2.0, 0.0, 0.0, 0.0 ],
                 [ -2.0, 0.0, 6 ** 0.5, 0.0, 0.0 ],
                 [ 0.0, -6.0 ** 0.5, 0.0, 6 ** 0.5, 0.0 ],
                 [ 0.0, 0.0, -6.0 ** 0.5, 0.0, 2.0 ],
                 [ 0.0, 0.0, 0.0, -2.0, 0.0 ] ], dtype = sp.complex128 )

sz = sp.array( [ [ 2.0, 0.0, 0.0, 0.0, 0.0 ],
                 [ 0.0, 1.0, 0.0, 0.0, 0.0 ],
                 [ 0.0, 0.0, 0.0, 0.0, 0.0 ],
                 [ 0.0, 0.0, 0.0, -1.0, 0.0 ],
                 [ 0.0, 0.0, 0.0, 0.0, -2.0 ] ] )

S_p = ( sx + 1.0j * sy ).real
S_n = ( sx - 1.0j * sy ).real


Hermitonian = lambda delta : 0.5 * ( sp.kron( S_p, S_n ) + sp.kron( S_n, S_p ) ) + delta * sp.kron( sz, sz )


def groundState( hermitonian, bondDim, timeIssue = None, wavefunc = None, out = None ):

    physicalDim = 5

    if not timeIssue:
        timeIssue = ( 0.05, 5.0 )

    if wavefunc:
        gamma, l = wavefunc
    else:
        gamma, l = sp.randn( 2, bondDim, 5, bondDim ), sp.randn( 2, bondDim )

    U = slg.expm( -timeIssue[ 0 ] * hermitonian )

    t = 0
    while t <= timeIssue[ 1 ]:
        if out:
            S_a, S_b = -sum( l[ 0 ] ** 2  * sp.log( l[ 0 ] ** 2 ) ), -sum( l[ 1 ] ** 2  * sp.log( l[ 1 ] ** 2 ) )
            expSz, expE = expValue( gamma, l, sz ), energy( gamma, l, hermitonian )
            print( "  {0:^+10.6f}{1:^+25.16f}{2:^+25.16f}{3:^+25.16f}{4:^+25.16f}".format( t, expE, expSz, S_a, S_b ), file = out )

        try:
            for A in range( 2 ):
                B = ( A + 1 ) % 2
                theta = sp.tensordot( l[ B ].reshape( bondDim, 1, 1 ) * gamma[ A ] * l[ A ], gamma[ B ] * l[ B ], 1 )
                theta = sp.tensordot( theta.reshape( bondDim, physicalDim ** 2, bondDim ), U, ( 1, 1 ) ).swapaxes( 1, 2 )
 
                X, Y, Z = slg.svd( theta.reshape( physicalDim * bondDim, physicalDim * bondDim ) )

                l[ A ] = Y[ 0:bondDim ] / slg.norm( Y[ 0:bondDim ] )
                gamma[ A ] = X[ :, 0:bondDim ].reshape( bondDim, 5, bondDim ) / l[ B ].reshape( bondDim, 1, 1 )
                gamma[ B ] = Z[ 0:bondDim, : ].reshape( bondDim, 5, bondDim ) / l[ B ]

            t += timeIssue[ 0 ]
        except slg.LinAlgError:
            print( "raise an exception", file = stderr )
            gamma += sp.random.normal( size = ( 2, bondDim, 5, bondDim ) )
            t = 0.0


    return gamma, l


def energy( g, l, hermitonian ):
    bondDim = l.shape[ 1 ]
    physicalDim = g.shape[ 2 ]

    energys = []
    for A in ( 0, 1 ):
        B = ( A + 1 ) % 2
        theta = sp.tensordot( l[ B ].reshape( bondDim, 1, 1 ) * g[ A ] * l[ A ], g[ B ] * l[ B ], 1 )
        Htheta = sp.tensordot( theta.reshape( bondDim, physicalDim ** 2, bondDim ), hermitonian, ( 1, 1 ) )
        energys.append( sp.tensordot( theta.reshape( bondDim, physicalDim ** 2, bondDim ), Htheta, ( [ 0, 1, 2 ], [ 0, 2, 1 ] ) ) )

    return sum( energys ) / 2.0


def expValue( g, l, op ):
    bondDim = l.shape[ 1 ]

    expVals = []
    for A in ( 0, 1 ):
        B = ( A + 1 ) % 2
        theta = l[ B ].reshape( bondDim, 1, 1 ) * g[ A ] * l[ A ]
        OpTheta = sp.tensordot( theta, op, ( 1, 1 ) )
        expVals.append( sp.tensordot( theta, OpTheta, ( [ 0, 1, 2 ], [ 0, 2, 1 ] ) ) )

    return sum( expVals ) / 2.0



if __name__ == '__main__':
    arg = docopt( __doc__ )

    parameter = { 'DIMBOUND' : 10, 'DELTA' : 1.0, 'TIMESTEP' : [ 0.05 ], 'TIME' : [ 5.0 ] }

    if arg[ 'waveInfo' ]:
        output = open( arg[ '-o' ], 'w' ) if arg[ '-o' ] else stdout
        waveFile = arg[ '<waveFile>' ]
        parameter[ 'DELTA' ] = float( arg[ '--delta' ] )

        try:
            f = sp.load( waveFile, 'r' )
            parameter[ 'DIMBOUND' ] = f[ 'l' ].shape[ 1 ]
            H = Hermitonian( parameter[ 'DELTA' ] )

            S_a, S_b = -sum( f[ 'l' ][ 0 ] ** 2  * sp.log( f[ 'l' ][ 0 ] ** 2 ) ), -sum( f[ 'l' ][ 1 ] ** 2  * sp.log( f[ 'l' ][ 1 ] ** 2 ) )
            expSz, expE = expValue( f[ 'gamma' ], f[ 'l' ], sz ), energy( f[ 'gamma' ], f[ 'l' ], H )

            print( "\n# bound dimension: {0}".format( parameter[ 'DIMBOUND' ] ), file = output )
            print( "# {0:^15}{1:^25}{2:^25}{3:^25}{4:^25}".format( "delta", "energy", "<Sz>", "entropy(A)", "entropy(B)" ), file = output )
            print( "  {0:^+15.6f}{1:^+25.16f}{2:^+25.16f}{3:^+25.16f}{4:^+25.16f}".format( parameter[ 'DELTA' ],
                expE, expSz, S_a, S_b ), file = output )

            f.close()
        except IOError:
            print( "There is no file {0}".format( waveFile ) )

        exit( 0 )

    parameterFile = arg[ '-f' ] if arg[ '-f' ] else 'INPUT'
    with open( parameterFile, 'r' ) as fileInput:
        for line in fileInput:
            p = line.strip()
            if p and p[ 0 ] != '#':
                p = p.replace( ' ', '' ).split( '=' )
                if p[ 0 ] == 'DIMBOUND':
                    parameter[ p[ 0 ] ] = int( p[ 1 ] )
                elif p[ 0 ] == 'TIMESTEP' or p[ 0 ] == 'TIME':
                    parameter[ p[ 0 ] ] = p[ 1 ].split( ',' )
                else:
                    parameter[ p[ 0 ] ] = float( p[ 1 ] )

    if len( parameter[ 'TIMESTEP' ] ) != len( parameter[ 'TIME' ] ):
        print( "The number of time step must match the number of time!" )
        exit( 1 )

    evolveTimes = list( zip( parameter[ 'TIMESTEP' ], parameter[ 'TIME' ] ) )
    H = Hermitonian( parameter[ 'DELTA' ] )

    if arg[ 'contin' ]:
        waveFile = arg[ '<waveFile>' ]

        try:
            f = sp.load( waveFile, 'r' )
            parameter[ 'DIMBOUND' ] = f[ 'l' ].shape[ 1 ]
            wf = ( f[ 'gamma' ], f[ 'l' ] )
            f.close()

            i = 1
            for timeIssue in evolveTimes:
                start_time = time()

                if arg[ '--noOutput' ]:
                    output = None
                else:
                    if timeIssue[ 0 ][ -1 ] == 'o':
                        output = open( arg[ '-o' ] + '-' + str( i ), 'w' ) if arg[ '-o' ] else stdout
                        print( "# time evolution: {0}".format( timeIssue ), file = output )
                        print( "# bound dimension: {0}".format( parameter[ 'DIMBOUND' ] ), file = output )
                        print( "# delta: {0}".format( parameter[ 'DELTA' ] ), file = output )
                        print( "\n\n# {0:^10}{1:^25}{2:^25}{3:^25}{4:^25}".format( "time", "energy", "<Sz>", "entropy(A)", "entropy(B)" ),
                            file = output )
                    else:
                        output = None

                stepT = float( timeIssue[ 0 ][ 0:-1 ] ) if timeIssue[ 0 ][ -1 ] == 'o' else float( timeIssue[ 0 ] )
                totalT = float( timeIssue[ 1 ] )
                wf = groundState( H, parameter[ 'DIMBOUND' ], ( stepT, totalT ), wf, output )
                sp.savez( waveFile + '_out' + '-' + str( i ), gamma = wf[ 0 ], l = wf[ 1 ] )

                i += 1
                if output:
                    print( "\n\n# running time: {0:.6f} (s)".format( time() - start_time ), file = output )

                    if arg[ '-o' ]:
                        output.close()

        except IOError:
            print( "There is no file {0}".format( wavefile ) )
            exit( 1 )
        except KeyboardInterrupt:
            print( "end of program!" )
            exit( 1 )
    else:
        waveFile = arg[ '-w' ] if arg[ '-w' ] else None

        try:
            wf = None
            i = 1
            for timeIssue in evolveTimes:
                start_time = time()

                if arg[ '--noOutput' ]:
                    output = None
                else:
                    if timeIssue[ 0 ][ -1 ] == 'o':
                        output = open( arg[ '-o' ] + '-' + str( i ), 'w' ) if arg[ '-o' ] else stdout
                        print( "# time evolution: {0}".format( timeIssue ), file = output )
                        print( "# bound dimension: {0}".format( parameter[ 'DIMBOUND' ] ), file = output )
                        print( "# delta: {0}".format( parameter[ 'DELTA' ] ), file = output )
                        print( "\n\n# {0:^10}{1:^25}{2:^25}{3:^25}{4:^25}".format( "time", "energy", "<Sz>", "entropy(A)", "entropy(B)" ),
                            file = output )
                    else:
                        output = None

                stepT = float( timeIssue[ 0 ][ 0:-1 ] ) if timeIssue[ 0 ][ -1 ] == 'o' else float( timeIssue[ 0 ] )
                totalT = float( timeIssue[ 1 ] )
                wf = groundState( H, parameter[ 'DIMBOUND' ], ( stepT, totalT ), wf, output )

                if waveFile:
                    sp.savez( waveFile + '_out' + '-' + str( i ), gamma = wf[ 0 ], l = wf[ 1 ] )

                i += 1
                if output:
                    print( "\n\n# running time: {0:.6f} (s)".format( time() - start_time ), file = output )

                    if arg[ '-o' ]:
                        output.close()

        except KeyboardInterrupt:
            print( "end of program!" )
            exit( 1 )

    exit( 0 )

