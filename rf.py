import math

f = float(input("Enter TX frequency in MHz: ") or 915)
f0 = f * 1e6
c = 299792458
lambda_ = c / f0

# TX power as measured at output SMA. Contains losses from SWGs, coupler, PCB, SMA
Ptx = float(input("Enter TX power in dBm: ") or 30.5)
# Power lost from the coaxial cable insertion loss
Lcbltx = float(input("Enter TX path cable loss in dB: ") or 0.2)
# Power lost from the antenna board's SMA connector insertion loss
Lsmatx = float(input("Enter TX path SMA loss in dB: ") or 0.145)
# VSWR of the TX antenna
vswr = float(input("Enter TX antenna VSWR: ") or 2)
while vswr < 1:
    print("VSWR must be greater than or equal to 1")
    vswr = float(input("Enter TX antenna VSWR: ") or 2)
# Calculation for the reflection coefficient from the VSWR
gamma = abs((vswr - 1) / (vswr + 1))
# Power lost from the TX antenna mismatch
Lmatch = -10 * math.log10(1 - gamma**2)
# Gain of the TX antenna in dBi
Gtx = float(input("Enter TX antenna gain in dBi: ") or 2.51)
# Total TX path loss in dB
Ltx = Lcbltx + Lsmatx + Lmatch
# Total TX power at the antenna in dBm
eirp = Ptx - Ltx + Gtx
print(f"Total TX power at the antenna: {eirp:.2f} dBm")

# Distance from the transmitter to the receiver in meters
d = float(input("Enter distance between TX and RX in meters: ") or 100000)

# Free space path loss in dB
Lfspl = 20 * math.log10(4 * math.pi * d / lambda_)
print(f"Free space path loss: {Lfspl:.2f} dB")

# Received power at the RX antenna in dBm
Prx = eirp - Lfspl
print(f"Received power at the RX antenna: {Prx:.2f} dBm")
# Gain of the RX antenna in dBi
Grx = float(input("Enter RX antenna gain in dBi: ") or 17.5)
# Power lost from the coaxial cable insertion loss
Lcblrx = float(input("Enter RX path cable loss in dB: ") or 5)
# Power lost from the antenna SMA connector insertion loss
Lsmarx = float(input("Enter RX path SMA loss in dB: ") or 0.145)
# Power lost from the RX antenna polarization mismatch in dB
Lpol = 3
# Total RX path loss in dB
Lrx = Lcblrx + Lsmarx + Lpol
# Total RX power at the receiver in dBm
Prx_total = Prx + Grx - Lrx
print(f"Total RX power at the receiver: {Prx_total:.2f} dBm")