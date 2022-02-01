#!/usr/bin/env python

"""
Tensorflow GA controller - New GA code which can simultaneously handle
continuous and discrete variables as genes


Note: this is not a generalized code and would require some work
      to work with a different set of input variables
"""

import os
from os import chdir
from os.path import exists
import pickle as pkl
import numpy as np
from random import randint, uniform
from subprocess import Popen, PIPE
from sys import argv
from math import floor, ceil, isnan
from decimal import Decimal

CWD = os.getcwd()

script, JOBNAME = argv
if '.' in JOBNAME:
    JOBNAME = JOBNAME.split('.')[0]

# ==============================================================
# SETUP THE GLOBAL PARAMETERS FOR THE CODE:
# ==============================================================

# --------------------------------------------------------------
# Modifying paramaters: Those you can edit with an input file

# Set up the hidden layer sizes
LAYERS = 4
LAYS = [1, 20] # Range of allowed variables for the layer sizes
NON_LINEAR = [0, 8]
CLASS  = False
OUTPUT = ['Purity']
# --------------------------------------------------------------

# Update the global variables if an input file exists:
if exists('{}.gain'.format(JOBNAME)):
    indats = open('{}.gain'.format(JOBNAME), 'r').readlines()
    for line in indats:
        if 'LAYERS' in line.upper():
            try:
                LAYERS = int(line.split('=')[1].strip())
            except ValueError:
                pass
        if 'OUTPUT' in line.upper():
            OUTPUT = line.split('=')[1].strip().split(',')
        if 'LINEAR' in line.upper():
            NON_LINEAR = [int(val) for val in line.split('=')[1].strip().split(',')]
        if 'CLASSIFICATION' in line.upper():
            if 'FALSE' in line.upper():
                CLASS = False
            elif 'TRUE' in line.upper():
                CLASS = True


# Set up the scaling parameters based on the number of input
# variables for the code - This inclue input AND output
NSCALE = ['b0_c', 'd0_c', 'q1_c', 'q2_c', 'U1_c', 'U2_c', 'b0_n',
          'd0_n', 'q1_n', 'q2_n', 'U1_n', 'U2_n', 'StructuredDensity',
          'InletTemp', 'tads', 'tblow', 'tevac', 'Pint', 'Plow', 'v0']
NSCALE += OUTPUT
#OUTPUT = ['Purity']

SCALES = [0, 4]
SCALE_LABEL = {0: 'none', 1: 'basic', 2: 'min-max', 3: 'mean', 4: 'stand'}
OPTIMIZERS  = {0: 'GradientDescent', 1: 'Adam'}
NONLINEARS  = {0: 'relu', 1: 'sigmoid', 2: 'tanh', 3: 'relu6', 4: 'crelu',
               5: 'selu', 6: 'elu', 7: 'softplus', 8: 'softsign',
               9: 'bias_add', 10: 'dropout'}

# The gene parameters are split into two categories:
# 1) DISC -> Discrete variables
# 2) CONT -> Continuous Variables
DISC = ['optimizer', 'nonlinear', 'minibatch_size']
CONT = ['learning_rate']
ALLGENES = DISC + CONT

# Define the allowed ranges and values for the gene parameters
VALUES = {'optimizer': [0, 1], 'nonlinear': NON_LINEAR,
          'minibatch_size': [50, 1000]}
RANGES = {'learning_rate': [-10., 0.]} # learning_rate: These are powers to the 10

# Other Control Parameters
OPTIONS = {'npop': 100, 'maxgen': 100, 'jobname': JOBNAME,
           'resubs': 3, 'Elite_Rate': 0.20, 'Mut_Rate': 0.2,
           'mingen': 20, 'converged': 10, 'Extinction_Rate': 0.15}
# ==============================================================
# For Tuning the Code:
O_OPTS = {'npop': 100, 'maxgen': 500, 'jobname': JOBNAME,
          'resubs': 3, 'Elite_Rate': 0.10, 'Mut_Rate': 0.2,
          'mingen': 100, 'converged': 40, 'Extinction_Rate': 0.15}
OVERRIDE = True
# ==============================================================
RESULTS = {}
STATS   = {}


def mkdir(name):
    try:
        os.mkdir(name)
    except OSError:
        pass
    except WindowsError:
        pass


def make_chromo(chromo):
    """Makes the label for the chromosome"""
    label = ''
    for gene in sorted(chromo):
        label += '{}:{}::'.format(gene, chromo[gene])
    return label


def define_layers():
    """Appends the input variables (genes) to correspond with the
    correct number of layers and the scaling parameters
    """
    global DISC, VALUES
    for layer in range(0, LAYERS):
        label = 'Layer_{}'.format(layer)
        DISC.append(label)
        VALUES[label] = LAYS
    for scale in NSCALE:
        label = 'Scale_{}'.format(scale)
        DISC.append(label)
        VALUES[label] = SCALES


def make_first_gen():
    """Makes the first generation"""
    gen, size, used = {}, 0, []
    while size < OPTIONS['npop']:
        gene = {}
#        label = ''
        # Start with the discrete variables:
        for idx, val in enumerate(DISC):
            low, hi = VALUES[val][0], VALUES[val][1]
            getit = randint(low, hi)
            gene[val] = getit
#            label += '{}:{}::'.format(val, getit)
        # Move on to the continuous variables"
        for idx, val in enumerate(CONT):
            low, hi = RANGES[val][0], RANGES[val][1]
            if val == 'learning_rate':
                getit = uniform(1.0, 9.9)
                power = randint(low, hi)
                getit = getit * (10 ** power)
            else:
                getit = uniform(low, hi)
            gene[val] = getit
#            label += '{}:{}::'.format(val, getit)
        label = make_chromo(gene)
        if label not in used:
            used.append(label)
            gen[label] = gene
            size += 1
    return gen


def run_gen(genes, gene_ids, id):
    """Sets up and runs the first generation"""
    mkdir('Gen_{}'.format(id))
    os.chdir('Gen_{}'.format(id))
    job_ids = []
    gene_ids['Gen_{}'.format(id)] = []
    sub_cwd = os.getcwd()
    for indx, gene in enumerate(sorted(genes)):
        x_cols = []
        y_cols = []
        layers = []
        gene_ids['Gen_{}'.format(id)].append(gene)
        for i in range(0, LAYERS):
            layers.append(0)
        mkdir('tf-CPU_{}'.format(indx))
        os.chdir('tf-CPU_{}'.format(indx))
        if gene in RESULTS:
            out = open('result.ga', 'w')
            out.write('done,{}'.format(1 - RESULTS[gene]['fit']))
            out.close()
            chdir(sub_cwd)
            continue
        tf_sub = 'tensor_net -o silent=True'
        for item in genes[gene]:
            if 'Scale' in item:
                var = item.split('Scale_')[1]
                type = SCALE_LABEL[int(genes[gene][item])]
                if var in OUTPUT:
                    if not CLASS:
                        if 'none' in type:
                            y_cols.append('{}'.format(var))
                        else:
                            y_cols.append('{}_{}'.format(var, type))
                    else:
                        y_cols.append('{}'.format(var))
                else:
                    if 'none' in type:
                        x_cols.append('{}'.format(var))
                    else:
                        x_cols.append('{}_{}'.format(var, type))
            elif 'optimizer' in item:
                optim = OPTIMIZERS[int(genes[gene][item])]
                tf_sub += ' -o optimizer={}'.format(optim)
            elif 'nonlinear' in item:
                nline = NONLINEARS[int(genes[gene][item])]
                tf_sub += ' -o non-linear={}'.format(nline)
            elif 'Layer_' in item:
                l_idx = int(item.split('Layer_')[1])
                neurons = genes[gene][item]
                layers[l_idx] = neurons
            else:
                tf_sub += ' -o {}={}'.format(item, genes[gene][item])
        tf_sub += ' -o x_cols='
        for idx, col in enumerate(x_cols):
            if idx == 0:
                tf_sub += '{}'.format(col)
            else:
                tf_sub += ',{}'.format(col)
        tf_sub += ' -o y_cols='
        for idx, col in enumerate(y_cols):
            if idx == 0:
                tf_sub += '{}'.format(col)
            else:
                tf_sub += ',{}'.format(col)
        tf_sub += ' -o layers={} -o vector_size='.format(LAYERS + 1)
        for idx, lay in enumerate(layers):
            if idx == 0:
                tf_sub += '{}'.format(lay)
            else:
                tf_sub += ',{}'.format(lay)
        tf_sub += ',{} {}.csv'.format(len(OUTPUT), OPTIONS['jobname'])

#        print tf_sub
        out = open('sub', 'w')
        out.write(tf_sub)
        out.close()
        make_link = Popen(['ln', '-s', '{}/{}.csv'.format(CWD, OPTIONS['jobname']), '.'])
        make_link.wait()
        make_exe = Popen(['chmod', '+x', 'sub'])
        make_exe.wait()

#        os.system('./sub')
        #print 'TF_CPU-{}'.format(indx)
        submit = Popen(['tfga-submit', '-N', 'TF_CPU-{}.{}'.format(indx, JOBNAME),
                        './sub'], stdout=PIPE)
        for line in submit.stdout:
            if 'Your job' in line:
                job_ids.append(int(line.split()[2]))

        os.chdir(sub_cwd)
    os.chdir(CWD)
    return job_ids, gene_ids


def import_state():
    """Imports the current state of the GA"""
    global OPTIONS, STATS, RESULTS
    if exists('{}.tfga'.format(OPTIONS['jobname'])):
        data = open('{}.tfga'.format(OPTIONS['jobname']), 'rb')
        state = pkl.load(data)
        data.close()
        OPTIONS = state['options']
        STATS   = state['stats']
        RESULTS = state['results']
    else:
        state = {'Gen': 0, 'Gene_ids': {}, 'resubbed': 0,
                 'options': OPTIONS, 'stats': {}}
    return state


def dump_state(state):
    """Creates a state file with the current state of the
    calculations
    """
    out = open('{}.tfga'.format(JOBNAME), 'wb')
    pkl.dump(state, out)
    out.close()


def check_lastgen(state):
    """Checks whether the last generation was successful"""
    global RESULTS
    done, lastg, failed = True, {}, []
    last = 'Gen_{}'.format(state['Gen'] - 1)
    chdir(last)
#    print " > Checking Results"
    for idx, gene in enumerate(state['Gene_ids'][last]):
        chdir('tf-CPU_{}'.format(idx))
        if not exists('result.ga'):
            if state['resubbed'] < OPTIONS['resubs']: 
                done = False
                failed.append(idx)
            else:
                if gene not in RESULTS:
                    RESULTS[gene] = {}
                    RESULTS[gene]['fit'] = 2.0
                    RESULTS[gene]['gen'] = state['Gen'] - 1
                    RESULTS[gene]['cpu'] = idx
            chdir('..')
            continue
        gadat = [val for val in open('result.ga', 'r').readlines()[0].split(',')][1]
        try:
            fit = 1 - float(gadat)
        except ValueError:
            fit = 2.0
        if isnan(fit):
            fit = 2.0
        if not np.isfinite(fit):
#            print idx, fit
            fit = 2.0
#        print "Saving Results..."
        if gene not in RESULTS:
            RESULTS[gene] = {}
            RESULTS[gene]['fit'] = fit
            RESULTS[gene]['gen'] = state['Gen'] - 1
            RESULTS[gene]['cpu'] = idx
#        print "Done"
        lastg[gene] = fit
        chdir('..')
    chdir('..')
#    print " > All Results From Previous Generation Saved.\n"
    return done, lastg, failed


def resubmit_genes(failed, state):
    """Resubmits failed runs"""
    last = 'Gen_{}'.format(state['Gen'] - 1)
    job_ids = []
    chdir(last)
    for fail in failed:
        chdir('tf-CPU_{}'.format(fail))
        submit = Popen(['tfga-submit', '-N', 're-TF_CPU-{}.{}'.format(fail, JOBNAME),
                        './sub'], stdout=PIPE)
        for line in submit.stdout:
            if 'Your job' in line:
                job_ids.append(int(line.split()[2]))
        chdir('..')
    chdir(CWD)
    return job_ids


def resubmit(job_ids, gen):
    """The main code resubmits itself behind the list of job_ids"""
    if len(job_ids) > 0:
        waitid = []
        for id in job_ids:
            waitid.append(str(id))
        sge_script = ['#!/bin/bash\n',
                      '#$ -cwd\n',
                      '#$ -V\n',
                      '#$ -j y\n',
                      '#$ -N tfga_Gen_{}.{}\n'.format(gen, JOBNAME),
                      '#$ -o {}.out\n'.format(JOBNAME),
                      '#$ -hold_jid %s\n' % ','.join(waitid),
                      'tfga ', JOBNAME]
    else:
        sge_script = ['#!/bin/bash\n',
                      '#$ -cwd\n',
                      '#$ -V\n',
                      '#$ -j y\n',
                      '#$ -N tfga_Gen_{}.{}\n'.format(gen, JOBNAME),
                      '#$ -o {}.out\n'.format(JOBNAME),
                      'tfga ', JOBNAME]
    sge_script = ''.join(sge_script)
    submit = Popen("qsub", shell=False, stdin=PIPE)
    submit.communicate(input=sge_script)
    exit()


def rank(pop):
    """Ranks the population and returns some stats"""
    ranks, directory, fits = {}, {}, []
    for gene in pop:
        fits.append(pop[gene])
        directory['%.6f' % pop[gene]] = gene
    fits = sorted(fits)
    for idx, fit in enumerate(fits):
        ranks['{}'.format(idx)] = {'gene': directory['%.6f' % fit], 'fit': fit}
    stats = {'best': min(fits), 'mean': np.mean(fits), 'stdev': np.std(fits)}
    return ranks, stats


def elitism(ranks):
    """Selects the best in the population for elitism"""
    genes, rate = {}, OPTIONS['Elite_Rate']
    sum = int(floor(OPTIONS['npop'] * rate))
    if sum < 2:
        sum = 2
    for i in range(0, sum):
        genes[ranks['{}'.format(i)]['gene']] = parse_genes(ranks['{}'.format(i)]['gene'])
    return genes, sum


def mating(elites, genes, ranks):
    """Performs the mating routines"""
    need = OPTIONS['npop'] - elites
    print "\n\n> Normalizing Set.."
    wheel, end = normalize(ranks)
    last = 0
    for rank in range(0, end):
#        print rank, wheel[str(rank)]['fit'], wheel[str(rank)]['fit'] - last
        last = wheel[str(rank)]['fit']
#    exit()
    print " > Done.\n> Starting Wheel"
    selected = []
    for kid in range(0, need):
        success = False
        # select mating pairs
#        print "  > Selecting Mating Pair..."
        while not success:
            mate1 = uniform(0., 1.0)
            mate2 = uniform(0., 1.0) + mate1
            t_mate1, t_mate2 = -1., -1.
#            print mate1, mate2
            if mate2 > 1.0:
                mate2 = mate2 - 1.0
            for i in range(0, OPTIONS['npop']):
                if mate1 < wheel['{}'.format(i)]['fit']:
                    t_mate1 = '{}'.format(i)
                    break
            for i in range(0, OPTIONS['npop']):
                if mate2 < wheel['{}'.format(i)]['fit']:
                    t_mate2 = '{}'.format(i)
                    break
#            print t_mate1, t_mate2
            if int(t_mate1) == int(t_mate2):
#                print "E000"
#                exit()
                continue
            if t_mate1 < 0 or t_mate2 < 0:
#                print "E001"
#                exit()
                continue
            pair = [int(t_mate1), int(t_mate2)]
            mates = '{}-{}'.format(min(pair), max(pair))
#            print "     > mates:", mates
            if mates not in selected:
                selected.append(mates)
                success = True
                #print len(selected), mates
#        print "  > Done.Staring Mating..."
        # combine the genes
        fit1, fit2 = wheel[t_mate1]['fit'], wheel[t_mate2]['fit']
        gene1, gene2 = wheel[t_mate1]['gene'], wheel[t_mate2]['gene']
        gene1 = parse_genes(gene1)
        gene2 = parse_genes(gene2)
        child = {}
        for chromo in gene1:
            if chromo in DISC:
                cut = uniform(0., 1.)
                par1 = (fit1 / (fit1 + fit2))
                if cut < par1:
                    cut = 0
                else:
                    cut = 1
                vals = [int(gene1[chromo]), int(gene2[chromo])]
                child[chromo] = str(vals[cut])
            else:
                vals = [float(gene1[chromo]), float(gene2[chromo])]
                top, bot = max(vals), min(vals)
#                h = uniform(0., 1.0)
                h = (fit1 / (fit1 + fit2))
                child[chromo] = str((h * vals[0]) + ((1 - h) * vals[1]))
        mutate = uniform(0.0, 1.0)
        if mutate < OPTIONS['Mut_Rate']:
#            print "     > Mutation!!!!"
            child = mutations(child)
#            print "     > Mutation Completed."
#        print "  > Child Generated from selected pair: {}\n".format(mates)
        child_tag = make_chromo(child)
        genes[child_tag] = child
    #print "Mating Done"
    return genes


def parse_genes(gene):
    """Takes in the gene's string a returns a dictionary"""
    gene = gene.split('::')[:-1]
    new = {}
    for item in gene:
        item = item.split(':')
        new[item[0]] = item[1]

    return new


def normalize(ranks):
    """Normalizes the population and builds the wheel"""
    sum, last = 0, 0
    wheel = {}
    extinct = 1. - OPTIONS['Extinction_Rate']
    end = int(floor(extinct * OPTIONS['npop']))
    if len(ranks) <= end:
        end = len(ranks) - 1
#    for rank in range(0, end):
#        print rank, (2 - ranks['{}'.format(rank)]['fit']) / 2
#    exit()
    for rank in range(0, end):
        sum += (2 - ranks['{}'.format(rank)]['fit'])
    print sum
    for rank in range(0, end):
        wheel['{}'.format(rank)] = ranks['{}'.format(rank)]
        wheel['{}'.format(rank)]['fit'] = ((2 - ranks['{}'.format(rank)]['fit']) / sum) + last
        last = wheel['{}'.format(rank)]['fit']
#    print "sum:", sum
    return wheel, end


def mutations(child):
    """Performs mutation"""
    which = randint(0, len(ALLGENES) - 1)
    gene  = ALLGENES[which]
#    print "       > {}'s Initial:".format(gene), child[gene]
    dir = randint(0, 1)
    if gene in DISC:
        range = VALUES[gene][1] - VALUES[gene][0]
        if range > 20:
            mute = int(floor(0.2 * range))
        else:
            mute = 1
        child[gene] = int(child[gene])
        if dir == 0:
            child[gene] -= mute
        if dir == 1:
            child[gene] += mute
        if child[gene] < VALUES[gene][0]:
            child[gene] = VALUES[gene][0]
        if child[gene] > VALUES[gene][1]:
            child[gene] = VALUES[gene][1]
    else:
        now = float(child[gene])
        x = uniform(0., 1.)
        if 'learning_rate' in gene:
            range = (float(RANGES[gene][1])) - (float(RANGES[gene][0]))
            base, power = [float(val) for val in str('%.2E' % Decimal(now)).split('E')]
            flux = uniform(0., 0.2) # Increases the randomness of the mutation
            if dir == 0:
                child[gene] = base * 10 ** (power - (flux * range * ((2 * x) - 1)))
            else:
                child[gene] = base * 10 ** (power + (flux * range * ((2 * x) - 1)))
            if child[gene] < 10 ** RANGES[gene][0]:
                child[gene] = 10 ** RANGES[gene][0]
            if child[gene] > 10 ** RANGES[gene][1]:
                child[gene] = 10 ** RANGES[gene][1]
        else:
            range = float(RANGES[gene][1]) - float(RANGES[gene][0])
            if dir == 1:
                child[gene] = now + (0.2 * range * ((2 * x) - 1))
            else:
                child[gene] = now - (0.2 * range * ((2 * x) - 1))
            if child[gene] < RANGES[gene][0]:
                child[gene] = RANGES[gene][0]
            if child[gene] > RANGES[gene][1]:
                child[gene] = RANGES[gene][1]
    child[gene] = str(child[gene])
#    print "       > range:", range
#    print "       > Becomes:", child[gene]
    return child


def update_results():
    """Makes the output file"""
    for_print, genes = {}, []
    for chromo in RESULTS:
        for_print[chromo] = parse_genes(chromo)
        for gene in for_print[chromo]:
            if gene not in genes:
                genes.append(gene)
            if 'Scale_' in gene:
                for_print[chromo][gene] = SCALE_LABEL[int(for_print[chromo][gene])]
            if 'nonlinear' in gene:
                for_print[chromo][gene] = NONLINEARS[int(for_print[chromo][gene])]
            if 'optimizer' in gene:
                for_print[chromo][gene] = OPTIMIZERS[int(for_print[chromo][gene])]
#        print for_print[chromo]
#        exit()
    genes = sorted(genes)
    out = open('{}/GA_Results.csv'.format(CWD), 'w')
    out.write('Generation,CPU,')
    out.write('fitness,Pearson_RSquared,')
    for gene in genes:
        out.write('{},'.format(gene))
    out.write('\n')
    for chromo in RESULTS:
        out.write('{},{},'.format(RESULTS[chromo]['gen'],
                                  RESULTS[chromo]['cpu']))
        out.write('{},{},'.format(RESULTS[chromo]['fit'],
                                  1 - RESULTS[chromo]['fit']))
        for gene in genes:
            try:
                out.write('{},'.format(for_print[chromo][gene]))
            except KeyError:
                out.write(',')
        out.write('\n')
    out.flush()
    out.close()


def update_stats():
    """Keeps a running update on the stats of the GA"""
    out = open('Runningstats_GA.csv', 'w')
    out.write('Gen,best,mean,stdev\n')
    for gen in sorted(STATS):
        gen_id = gen.split('Gen_')[1]
        out.write('{},{},{},{}\n'.format(gen_id, STATS[gen]['best'],
                                         STATS[gen]['mean'],
                                         STATS[gen]['stdev']))
    out.flush()
    out.close()


def convergence_check():
    """Checks whether the optimization has converged"""
    bests = []
    gens = len(STATS)
    for gen in range(gens - OPTIONS['converged'], gens):
       label = 'Gen_{}'.format(gen)
       bests.append(STATS[label]['best'])
#    for set in sorted(STATS):
#        bests.append(STATS[set]['best'])
#    bests = sorted(bests)
    print bests[-OPTIONS['converged']:]
#    print np.std(bests[-OPTIONS['converged']:])
    if np.std(bests[-OPTIONS['converged']:]) > 1e-15:
        print ("Convergence not met: " + str(np.std(bests[-OPTIONS['converged']:])))
        return False
    else:
        print ("Optimization has converged.")
        return True


def main():
    """Main Execution of the Script"""
    global RESULTS, STATS, OPTIONS
    define_layers()
    state, converged = import_state(), False
    if OVERRIDE:
        OPTIONS = O_OPTS

    # Generate the initial population and run the generation
    if state['Gen'] == 0:
        genes = make_first_gen()
        state['Gen'] += 1

    # Run subsequent generations
    else:
        job_ids = []
        done, last, failed = check_lastgen(state)
        if done or (state['resubbed'] >= OPTIONS['resubs'] and len(last) > 10):
            if len(failed) != 0:
                if exists('{}/failed_points.out'.format(CWD)):
                    failed_out = open('{}/failed_points.out'.format(CWD), 'a')
                else:
                    failed_out = open('{}/failed_points.out'.format(CWD), 'w')
                failed_out.write('{},{}\n'.format(state['Gen'] - 1, failed))
                failed_out.flush()
                failed_out.close()
            state['resubbed'] = 0
            pop_ranks, pop_stats = rank(last)
            genes, elites = elitism(pop_ranks)
            genes = mating(elites, genes, pop_ranks)
            STATS['Gen_{}'.format(state['Gen'] - 1)] = pop_stats
            state['stats'] = STATS
            if state['Gen'] - 1 > OPTIONS['mingen']:
                converged = convergence_check()
                if converged:
                    print ("\n\n" + "-" * 50)
                    print ("Optimization Converged Successfully After")
                    print (" {} Generations".format(state['Gen'] - 1))
                    print ("-" * 50 + '\n')
#                    exit()
            state['Gen'] += 1
            update_results()
            update_stats()
            if converged:
                exit()
        else:
            state['job_ids'] = resubmit_genes(failed, state)
            state['resubbed'] += 1
            dump_state(state)
            resubmit(state['job_ids'], state['Gen'] - 1)

    # Submit the genes and log the state of the calculation
    job_ids, state['Gene_ids'] = run_gen(genes, state['Gene_ids'], state['Gen'] - 1)
    state['job_ids'] = job_ids
    state['results'] = RESULTS

    # save the calculation state and resubmit the GA's parent code
    dump_state(state)
    resubmit(job_ids, state['Gen'] - 1)


if __name__ in '__main__':
    main()

