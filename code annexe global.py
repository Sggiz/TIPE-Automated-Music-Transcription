# Code principal

from time import time

timelog = {}
timelog["start"] = time()

from scipy.signal.windows import hann
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

import partitura as pt


timelog["import de modules"] = time()

file_name = "../../Matériel audio/Gamme.wav"

fs, raw_file = wavfile.read(file_name)
s = raw_file[:, 0]  # conversion en mono

timelog["import du fichier"] = time()

N = len(s)  # nombre de points
dt = 1/fs  # intervalle de temps entre mesures
t = np.arange(N) * dt

tmin, tmax = 0, 10  # s
imin, imax = int(tmin/dt), int(tmax/dt)
N_fenetre = imax - imin
s_fenetre = s[imin:imax]
t_fenetre = t[imin:imax]

timelog["fenetrage"] = time()


##########################
# Determination du tempo #
##########################

s_norm = s_fenetre/np.max(s_fenetre)
s_carre = s_norm * s_norm

fmin = 2
fc = 6  # en Hz, frequence de coupure du passe-bas

s_filtre = [0]
for n in range(N_fenetre - 1):
    v = s_filtre[n] + 2*np.pi*fc/fs*(s_carre[n] - s_filtre[n])
    s_filtre.append(v)

S_env = np.absolute(rfft(s_filtre))
freq_env = rfftfreq(N_fenetre, dt)

S_fenetre_env = S_env[int(fmin/freq_env[1]):int(fc/freq_env[1])]
freq_fenetre_env = freq_env[int(fmin/freq_env[1]):int(fc/freq_env[1])]

bps = freq_fenetre_env[np.argmax(S_fenetre_env)]

timelog['filtrage / fft enveloppe'] = time()


###########################
# Calcul du spectrogramme #
###########################

N_win = 2**13  # Nombre de points de la fenetre
hop = 2**10  # taille du saut de fenetre (en nombre d'echantillons/points)

win = hann(N_win)
n = (N_fenetre - N_win)//hop + 1  # nombre d'analyses
p = N_win//2 + 1  # nombre de points d'une transformation

S = np.zeros((p, n))

T = np.zeros(n)

df = fs/N_win
freq = rfftfreq(N_win, dt)

for j in range(n):
    # indices de début et de fin de la fenetre consideree
    i_s = j*hop
    i_e = i_s + N_win

    # l'instant de l'analyse (milieu de fenetre)
    _t = (i_s+i_e)/2 * dt

    # l'extrait analyse et le calcul de sa transformee
    _s = s_fenetre[i_s:i_e] * win
    _S = np.absolute(rfft(_s))

    # Ajustement Local ("normalisation")
    m = np.max(_S)
    if m > 0:
        _S /= m

    T[j] = _t
    for i in range(p):
        S[i, j] = _S[i]

timelog["spectrogramme"] = time()


##############################
# Recherche de la note jouee #
##############################

def somme_spect(S, f, eps):
    somme, k, i = 0, 1, round(f/df)
    while i < len(S):
        # fenetre de taille ~2eps servant a compenser le caractere
        # quasi-harmonique de l'instrument
        win = S[max(i-eps, 0):min(i+eps, len(S))]
        somme += np.max(win)
        k += 1
        i = round(k*f/df)
    return somme


Mel_freq = np.zeros(n)

prominence = 0.1  # prominence minimale des pics candidates

f_eps = 10  # taille spectrale de la fenetre pour la somme spectrale

i_eps = round(f_eps/df)
for i in range(n):
    _S = S[:, i]
    peaks, _ = find_peaks(_S, prominence=prominence)
    f_cand = freq[peaks]  # frequences candidates

    # recherche du meilleur candidat (argmax de somme spectrale)
    f_note, s_max = 0, 0
    for f in f_cand:
        somme = somme_spect(_S, f, i_eps)
        if somme > s_max:
            f_note = f
            s_max = somme
    Mel_freq[i] = f_note


def freq_to_note(f):
    if f == 0:
        return -1
    N_ref, f_ref = 69, 440
    N = N_ref + 12 * np.log2(f/f_ref)
    return round(N)


Mel = np.array([freq_to_note(f) for f in Mel_freq])

timelog["recherche melodie"] = time()


#######################
# Recherche des notes #
#######################

notes = []

delta_t_min = 0.125  # s
note_val, note_debut = -1, T[0]

for i, t in enumerate(T):
    if Mel[i] == note_val:
        continue

    if note_val != -1 and (t - note_debut >= delta_t_min) and note_val > 45:
        notes.append((note_val, note_debut, t))

    note_val = int(Mel[i])
    note_debut = t

if note_val != -1 and (t - note_debut >= delta_t_min) and note_val > 45:
        notes.append((note_val, note_debut, t))
    
timelog["recherche evenements"] = time()


############################
# Creation de la partition #
############################

#octave sous forme [(step,alter)] : ('C',1) -> C# et ('B',-2) -> B double flat
octave_notes = [('C',0), ('C',1), ('D',0), ('D',1), ('E',0), ('F',0),
                ('F',1), ('G',0), ('G',1), ('A',0), ('A',1), ('B',0)]
octave_aff = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
len_octave = 12  #12 demi-tons dans la gamme tempérée standard
octave_shift = -1  #alignement midi - numéro d'octave

def number_to_pitch(n):
    step, alter = octave_notes[n%len_octave]
    octave = n//len_octave + octave_shift
    return step, alter, octave

def number_to_aff(n):
    step12 = octave_aff[n%len_octave]
    octave = n//len_octave + octave_shift
    return step12 + str(octave)

start_time = min([t0 for _, t0, _ in notes])

quarter_duration = 4 #division d'un battement : 4 -> analyse à la double croche près
duration_length = 1 / bps / quarter_duration #durée d'une unité de temps

def time_to_duration(t):
    return round((t-start_time)/duration_length)

def duration_to_time(d):
    return start_time + d * duration_length

part = pt.score.Part(id='t00', part_name="test00", quarter_duration=quarter_duration)

for i, (n, t0, t1) in enumerate(notes):
    step, alter, octave = number_to_pitch(n)
    note = pt.score.Note(id=f"n{i}", step=step, octave=octave, alter=alter)
    
    start, end = time_to_duration(t0), time_to_duration(t1)
    part.add(note, start, end)

part.add(pt.score.TimeSignature(4, 4), start=0)
pt.score.add_measures(part)
pt.score.tie_notes(part)

pt.save_musicxml(part, "partition_gamme.xml")

timelog["creation partition"] = time()


#############
# Affichage #
#############

fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 6))

axs[0].plot(t_fenetre, s_fenetre)
axs[0].set_title("Extrait Audio")

axs[1].plot(T, Mel, ".")

for n, t0, t1 in notes:
    axs[2].hlines(n, t0, t1, color="red")
    axs[2].text(t0, n + 0.5, str(n), color="red")

for i in range(int((t_fenetre[-1] - start_time)/duration_length)):
    t = start_time + i*duration_length
    if i%quarter_duration == 0:
        axs[3].axvline(t, color="grey")
    else:
        axs[3].axvline(t, color="lightgrey")

for n, t0, t1 in notes:
    t_start = duration_to_time(time_to_duration(t0))
    t_end = duration_to_time(time_to_duration(t1))
    axs[3].hlines(n, t_start, t_end, color='blue')
    axs[3].text(t0, n + 0.5, number_to_aff(n), color="blue")
    

timelog['display'] = time()

deltaTimelog = {n: int((t-timelog["start"])*100)/100
                for (n, t) in timelog.items()}

print(deltaTimelog)

plt.show()
