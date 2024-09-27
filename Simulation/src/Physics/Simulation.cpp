#include "Simulation.h"
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include "Units.h"
#include "fftw3.h"
#include <random>

Simulation::Simulation()
{
    // Konstruktion der Module
    timeIntegration = std::make_shared<TimeIntegration>();
    dataManager = std::make_shared<DataManager>("../../output_data/");
    icDataReader = std::make_shared<ICDataReader>();
    console = std::make_shared<Console>();
}

Simulation::~Simulation(){}

// In Simulation.cpp
void Simulation::updateScaleFactor(double fixedStep)
{
    // Einfache Integration der Friedmann-Gleichung für ein einfaches Universum
    // Dies ist eine stark vereinfachte Version. Für realistischere Simulationen sollten komplexere Modelle verwendet werden.
    // Beispiel: a(t + dt) = a(t) + H(t) * a(t) * dt
    double H = H_initial / (a * a * a); // Beispielhafte Annahme
    a += H * a * fixedStep;
}

std::vector<std::complex<double>> Simulation::generateGaussianRandomField(int N, double boxSize, double amplitude)
{
    std::vector<std::complex<double>> field(N * N * N, 0.0);

    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0, 1.0);

    for(int i = 0; i < N; ++i)
    {
        int kx = (i <= N/2) ? i : i - N;
        for(int j = 0; j < N; ++j)
        {
            int ky = (j <= N/2) ? j : j - N;
            for(int k = 0; k < N; ++k)
            {
                int kz = (k <= N/2) ? k : k - N;
                int index = (i * N + j) * N + k;

                if(kx == 0 && ky == 0 && kz == 0) // DC-Komponente
                {
                    field[index] = 0.0;
                    continue;
                }

                // Power Spectrum (z.B. P(k) proportional zu k^n, n = 1 für scale-invariant)
                double dk = std::sqrt(kx * kx + ky * ky + kz * kz);
                double Pk = (dk > 0) ? amplitude / (dk * dk) : 0.0;

                double real = dist(gen) * std::sqrt(Pk / 2.0);
                double imag = dist(gen) * std::sqrt(Pk / 2.0);

                field[index] = std::complex<double>(real, imag);
            }
        }
    }

    return field;
}
bool Simulation::init()
{
    // Lade die Konfigurationsdatei
    if (!dataManager->loadConfig("../Config.ini", this))
    {
        std::cerr << "Error: Could not load the config file." << std::endl;
        return false;
    }
    
    fixedStep = endTime / fixedTimeSteps;

    std::cout << "Total Number of Particles in the Config.ini file: " << numberOfParticles << std::endl;

    // Fehlerüberprüfungen
    if (minTimeStep > maxTimeStep)
    {
        std::cerr << "Error: minTimeStep is greater than maxTimeStep." << std::endl;
        return false;
    }
    if (endTime / minTimeStep < fixedTimeSteps)
    {
        std::cerr << "Error: endTime / minTimeStep is smaller than fixedTimeSteps." << std::endl;
        return false;
    }

    // Allgemeine Informationen zur Simulation ausgeben
    std::cout << "\nSimulation parameters:" << std::endl;
    std::cout << "  Number of particles: " << numberOfParticles << std::endl;
    std::cout << "  End time: " << std::scientific << std::setprecision(0) << (double)endTime / (double)3600.0 << " years" << std::endl;

    // Rechnerinformationen ausgeben
    Console::printSystemInfo();

    // Initialisieren der PM-Parameter
    gridSize = 64; // Erhöhen Sie die Gittergröße für höhere Auflösung
    boxSize = 1.0e21; // Größe der Simulationsbox in Metern (angepasst je nach Bedarf)
    deltaX = boxSize / gridSize;

    int totalGridPoints = gridSize * gridSize * gridSize;
    densityGrid.resize(totalGridPoints, 0.0);
    std::cout << "Density grid size: " << densityGrid.size() << std::endl;
    potentialGrid.resize(totalGridPoints);
    std::cout << "Potential grid size: " << potentialGrid.size() << std::endl;
    forceGrid.resize(totalGridPoints, std::vector<double>(3, 0.0));
    std::cout << "Force grid size: " << forceGrid.size() << std::endl;

    // Sicherstellen, dass der Partikelvektor groß genug ist
    if (particles.size() < static_cast<size_t>(numberOfParticles)) {
        particles.resize(numberOfParticles);
    }

    // Zufallszahlengenerator für die Initialisierung
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, boxSize);

    // Definieren Sie den Skalierungsfaktor (a(t)) und die Hubble-Konstante (H(t))
    a = 1.0e-3; // Anfangsskalierungsfaktor (z.B. a = 1 / 1000 für frühzeitliches Universum)
    double H_initial = 70.0 * 1e3 / 3.0857e22; // Hubble-Konstante in s^-1 (z.B. 70 km/s/Mpc umgerechnet in SI-Einheiten)

    // Gesamtmasse festlegen (z.B. Masse einer Galaxie oder eines Clusters)
    double totalMass = 1.0e40; // Gesamtmasse in kg (anpassen je nach Simulationsgröße)
    double particleMass = totalMass / numberOfParticles;

    // Amplitude der Dichtefluktuationen
    double amplitude = 0.01; // Erhöhte Amplitude für signifikante Fluktuationen

    // Generieren Sie ein Gaussian Random Field (GRF) für realistische Dichtefluktuationen
    std::vector<std::complex<double>> delta_k = generateGaussianRandomField(gridSize, boxSize, amplitude);

    // Inverse FFT durchführen, um das Realraum-Dichtefeld zu erhalten
    fftw_complex *in_fft = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * gridSize * gridSize * gridSize));
    fftw_complex *out_fft = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * gridSize * gridSize * gridSize));

    for(int i = 0; i < gridSize * gridSize * gridSize; ++i)
    {
        in_fft[i][0] = delta_k[i].real();
        in_fft[i][1] = delta_k[i].imag();
    }

    fftw_plan plan = fftw_plan_dft_3d(gridSize, gridSize, gridSize, in_fft, out_fft, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Normieren und Skalieren des Dichtefeldes
    std::vector<double> delta_real(gridSize * gridSize * gridSize, 0.0);
    for(int i = 0; i < gridSize * gridSize * gridSize; ++i)
    {
        delta_real[i] = out_fft[i][0] / (gridSize * gridSize * gridSize); // Realteil
    }

    fftw_destroy_plan(plan);
    fftw_free(in_fft);
    fftw_free(out_fft);

    // Verwenden Sie delta_real zur Modulation der Teilchenmassen
    for(int i = 0; i < numberOfParticles; i++)
    {
        double x = dis(gen);
        double y = dis(gen);
        double z = dis(gen);

        // Finden Sie die entsprechenden Gitterpunkte
        int gi = static_cast<int>((x / boxSize) * gridSize) % gridSize;
        int gj = static_cast<int>((y / boxSize) * gridSize) % gridSize;
        int gk = static_cast<int>((z / boxSize) * gridSize) % gridSize;

        size_t index = (gi * gridSize + gj) * gridSize + gk;
        double delta = 1.0 + amplitude * delta_real[index]; // Modulation basierend auf Dichtefluktuation

        particles[i] = std::make_shared<Particle>();
        particles[i]->position = vec3(x, y, z); // Komovierende Positionen
        particles[i]->velocity = vec3(H_initial * a * x, 
                                      H_initial * a * y, 
                                      H_initial * a * z); // Hubble-Fluss
        particles[i]->mass = particleMass * delta; // Masse mit Fluktuationen
        particles[i]->type = 1;

        // Periodische Randbedingungen anwenden
        applyPeriodicBoundary(particles[i]);
    }

    // Gesamtimpuls berechnen und korrigieren (sollte Null sein)
    vec3 totalMomentum(0.0, 0.0, 0.0);
    double totalMassActual = 0.0;
    for (const auto& particle : particles)
    {
        totalMomentum += particle->velocity * particle->mass;
        totalMassActual += particle->mass;
    }

    vec3 correctionVelocity = totalMomentum / totalMassActual;

    for (auto& particle : particles)
    {
        particle->velocity -= correctionVelocity / particle->mass;
    }

    if(numberOfParticles != particles.size())
    {
        std::cerr << "Error: Number of particles in the simulation does not match the number of particles in the data file." << std::endl;
        return false;
    }

    // Überprüfen, ob alle Partikel initialisiert sind
    for (int i = 0; i < numberOfParticles; i++)
    {
        if (!particles[i]) 
        {
            std::cerr << "Error: Particle " << i << " is not initialized." << std::endl;
            return false;
        }
    }

    // Info-Datei schreiben und Daten speichern
    dataManager->writeInfoFile(fixedStep, fixedTimeSteps, numberOfParticles);
    dataManager->saveData(particles, 0);

    std::cout << "Simulation initialized." << std::endl;

    return true;
}

void Simulation::run()
{
    // Initialer Kick (halber Zeitschritt)
    for(int i = 0; i < numberOfParticles; i++)
    {
        timeIntegration->Kick(particles[i], fixedStep / 2.0);
    }

    for(int t = 0; t < fixedTimeSteps; t++)
    {
        // 1. Drift (ganzer Zeitschritt)
        for(int i = 0; i < numberOfParticles; i++)
        {
            timeIntegration->Drift(particles[i], fixedStep);
            // Periodische Randbedingungen
            applyPeriodicBoundary(particles[i]);
        }

        // 2. Kräfte berechnen
        assignMassToGrid();
        computePotential();
        computeForces();
        interpolateForcesToParticles();

        // 3. Kick (ganzer Zeitschritt)
        for(int i = 0; i < numberOfParticles; i++)
        {
            timeIntegration->Kick(particles[i], fixedStep);
        }

        // 4. Skalierungsfaktor aktualisieren
        updateScaleFactor(fixedStep);

        // 5. Daten speichern und Fortschritt anzeigen
        dataManager->saveData(particles, static_cast<int>(t) + 1);
        console->printProgress(static_cast<int>(t), fixedTimeSteps, "");
    }

    // Finaler Kick (halber Zeitschritt)
    for(int i = 0; i < numberOfParticles; i++)
    {
        timeIntegration->Kick(particles[i], fixedStep / 2.0);
    }

    std::cout << "Simulation finished." << std::endl;
}


void Simulation::applyPeriodicBoundary(std::shared_ptr<Particle> particle)
{
    if(particle->position.x < 0) particle->position.x += boxSize;
    if(particle->position.x >= boxSize) particle->position.x -= boxSize;

    if(particle->position.y < 0) particle->position.y += boxSize;
    if(particle->position.y >= boxSize) particle->position.y -= boxSize;

    if(particle->position.z < 0) particle->position.z += boxSize;
    if(particle->position.z >= boxSize) particle->position.z -= boxSize;
}

void Simulation::initGasParticleProperties(std::shared_ptr<Tree> tree)
{
    // Update der Eigenschaften der Gas-Partikel
    for (int i = 0; i < numberOfParticles; i++)
    {
        if(particles[i]->type == 2)
        {
            // Berechnung von U aus T: u = 1 / (gamma-1) * bk * T / (meanMolWeight * prtn)
            particles[i]->U = (1.0 / (Constants::GAMMA - 1.0)) * Constants::BK * particles[i]->T / (Constants::meanMolWeight * Constants::prtn);
            // Berechnung von P: P = (gamma-1) * u * rho
            particles[i]->P = (Constants::GAMMA - 1.0) * particles[i]->U * particles[i]->rho;
        }
    }

    // Berechnung des Median-Drucks
    tree->root->calcMedianPressure();
}

void Simulation::updateGasParticleProperties(std::shared_ptr<Tree> tree)
{
    // Update der Eigenschaften der Gas-Partikel
    for (int i = 0; i < numberOfParticles; i++)
    {
        if(particles[i]->type == 2)
        {
            // Berechnung von P: P = (gamma-1) * u * rho
            particles[i]->P = (Constants::GAMMA - 1.0) * particles[i]->U * particles[i]->rho;
        }
    }
    
    // Berechnung des Median-Drucks
    tree->root->calcMedianPressure();
}

void Simulation::calculateForcesWithoutOctree(std::shared_ptr<Particle> p)
{
    p->acceleration = vec3(0.0, 0.0, 0.0);
    p->dUdt = 0;

    for (int j = 0; j < numberOfParticles; j++)
    {
        if (p != particles[j])
        {
            vec3 d = particles[j]->position - p->position;
            double r = d.length();
            if (r > 0) // Vermeidung von Division durch Null
            {
                vec3 newAcceleration = d * (Constants::G * particles[j]->mass / (r * r * r));
                p->acceleration += newAcceleration;
            }
        }
    }
}

void Simulation::assignMassToGrid()
{
    // Alle Gitterzellen auf Null setzen
    std::fill(densityGrid.begin(), densityGrid.end(), 0.0);

    for (const auto& particle : particles)
    {
        // Anpassung der Positionen innerhalb [0, boxSize)
        double posX = fmod(particle->position.x, boxSize);
        if (posX < 0) posX += boxSize;
        double posY = fmod(particle->position.y, boxSize);
        if (posY < 0) posY += boxSize;
        double posZ = fmod(particle->position.z, boxSize);
        if (posZ < 0) posZ += boxSize;

        // Normierte Positionen im Bereich [0, gridSize)
        double x = (posX / boxSize) * gridSize;
        double y = (posY / boxSize) * gridSize;
        double z = (posZ / boxSize) * gridSize;

        int i = static_cast<int>(std::floor(x));
        int j = static_cast<int>(std::floor(y));
        int k = static_cast<int>(std::floor(z));

        // Sichere Modulo-Anwendung
        i = safeMod(i, gridSize);
        j = safeMod(j, gridSize);
        k = safeMod(k, gridSize);

        double dx = x - std::floor(x);
        double dy = y - std::floor(y);
        double dz = z - std::floor(z);

        // Cloud-in-Cell (CIC) Gewichte berechnen
        double wx[2] = {1.0 - dx, dx};
        double wy[2] = {1.0 - dy, dy};
        double wz[2] = {1.0 - dz, dz};

        // Masse auf die 8 umliegenden Gitterpunkte verteilen
        for (int ii = 0; ii <= 1; ++ii)
        {
            int gi = safeMod(i + ii, gridSize);
            for (int jj = 0; jj <= 1; ++jj)
            {
                int gj = safeMod(j + jj, gridSize);
                for (int kk = 0; kk <= 1; ++kk)
                {
                    int gk = safeMod(k + kk, gridSize);
                    size_t index = static_cast<size_t>(gi * gridSize * gridSize + gj * gridSize + gk);
                    double weight = wx[ii] * wy[jj] * wz[kk];
                    densityGrid[index] += particle->mass * weight;
                }
            }
        }
    }
}
void Simulation::computePotential()
{
    // Dichtekontraste berechnen
    double meanDensity = std::accumulate(densityGrid.begin(), densityGrid.end(), 0.0) / densityGrid.size();
    for (auto& rho : densityGrid)
    {
        rho = (rho - meanDensity) / meanDensity;
    }

    // Vorbereitung für FFTW
    int N = gridSize;
    fftw_complex *in = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N * N * N));
    fftw_complex *out = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N * N * N));

    // Dichte in das Eingabefeld kopieren
    for (int i = 0; i < N * N * N; ++i)
    {
        in[i][0] = densityGrid[i]; // Realteil
        in[i][1] = 0.0;            // Imaginärteil
    }

    // Vorwärtstransformation
    fftw_plan plan_forward = fftw_plan_dft_3d(N, N, N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan_forward);

    // Lösen der Poisson-Gleichung im Fourier-Raum
    double G = 6.67430e-11; // Gravitationskonstante in m^3 kg^-1 s^-2
    for (int i = 0; i < N; ++i)
    {
        int kx = (i <= N / 2) ? i : i - N;
        for (int j = 0; j < N; ++j)
        {
            int ky = (j <= N / 2) ? j : j - N;
            for (int l = 0; l < N; ++l) // Schleifenvariable von k zu l
            {
                int kz = (l <= N / 2) ? l : l - N;
                int index = (i * N + j) * N + l;
                double k_squared = (kx * kx + ky * ky + kz * kz) * (2 * M_PI / boxSize) * (2 * M_PI / boxSize);
                if (k_squared != 0)
                {
                    double factor = -4.0 * M_PI * G / k_squared;
                    out[index][0] *= factor;
                    out[index][1] *= factor;
                }
                else
                {
                    out[index][0] = 0.0;
                    out[index][1] = 0.0;
                }
            }
        }
    }

    // Rücktransformation
    fftw_plan plan_backward = fftw_plan_dft_3d(N, N, N, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan_backward);

    // Potenzial auslesen und normalisieren
    for (int i = 0; i < N * N * N; ++i)
    {
        potentialGrid[i] = std::complex<double>(in[i][0] / (N * N * N), in[i][1] / (N * N * N));
    }

    // Speicher freigeben
    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);
    fftw_free(in);
    fftw_free(out);
}

void Simulation::computeForces()
{
    int N = gridSize;
    double factor = 1.0 / (deltaX * N); // Skalierungsfaktor für die Ableitung

    // Kräftefeld zurücksetzen
    for (auto& f : forceGrid)
    {
        std::fill(f.begin(), f.end(), 0.0);
    }

    // Vorbereitung für FFTW
    fftw_complex *in = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N * N * N));
    fftw_complex *out = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N * N * N));

    // Potenzial in das Eingabefeld kopieren (Realraum)
    for (int i = 0; i < N * N * N; ++i)
    {
        in[i][0] = potentialGrid[i].real();
        in[i][1] = potentialGrid[i].imag();
    }

    // Vorwärtstransformation des Potenzials in den Fourier-Raum
    fftw_plan plan_forward = fftw_plan_dft_3d(N, N, N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan_forward);

    for (int dir = 0; dir < 3; ++dir)
    {
        // Kopieren der Fourier-transformierten Potenzialdaten
        for (int idx = 0; idx < N * N * N; ++idx)
        {
            in[idx][0] = out[idx][0];
            in[idx][1] = out[idx][1];
        }

        // Multiplikation mit ik im Fourier-Raum
        for (int i = 0; i < N; ++i)
        {
            int kx = (i <= N / 2) ? i : i - N;
            for (int j = 0; j < N; ++j)
            {
                int ky = (j <= N / 2) ? j : j - N;
                for (int l = 0; l < N; ++l) // Schleifenvariable von k zu l
                {
                    int kz = (l <= N / 2) ? l : l - N;
                    int index = (i * N + j) * N + l;

                    int kdir;
                    if (dir == 0)
                        kdir = kx;
                    else if (dir == 1)
                        kdir = ky;
                    else
                        kdir = kz;

                    double k_factor = (2 * M_PI / boxSize) * kdir;

                    double real = in[index][0];
                    double imag = in[index][1];

                    // Ableitung im Fourier-Raum: Multiplikation mit ik
                    in[index][0] = -imag * k_factor;
                    in[index][1] = real * k_factor;
                }
            }
        }

        // Rücktransformation in den Realraum
        fftw_plan plan_backward = fftw_plan_dft_3d(N, N, N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(plan_backward);

        // Kräfte auslesen und skalieren
        for (int idx = 0; idx < N * N * N; ++idx)
        {
            forceGrid[idx][dir] = (out[idx][0] / (N * N * N)) * factor;
        }

        fftw_destroy_plan(plan_backward);
    }

    fftw_destroy_plan(plan_forward);
    fftw_free(in);
    fftw_free(out);
}

inline int Simulation::safeMod(int a, int b)
{
    int result = a % b;
    if (result < 0)
        result += b;
    return result;
}

void Simulation::interpolateForcesToParticles()
{
    for (auto& particle : particles)
    {
        // Anpassung der Positionen innerhalb [0, boxSize)
        double posX = fmod(particle->position.x, boxSize);
        if (posX < 0) posX += boxSize;
        double posY = fmod(particle->position.y, boxSize);
        if (posY < 0) posY += boxSize;
        double posZ = fmod(particle->position.z, boxSize);
        if (posZ < 0) posZ += boxSize;

        // Normierte Positionen im Bereich [0, gridSize)
        double x = (posX / boxSize) * gridSize;
        double y = (posY / boxSize) * gridSize;
        double z = (posZ / boxSize) * gridSize;

        int i = static_cast<int>(std::floor(x));
        int j = static_cast<int>(std::floor(y));
        int k = static_cast<int>(std::floor(z));

        // Sichere Modulo-Anwendung
        i = safeMod(i, gridSize);
        j = safeMod(j, gridSize);
        k = safeMod(k, gridSize);

        double dx = x - std::floor(x);
        double dy = y - std::floor(y);
        double dz = z - std::floor(z);

        // Cloud-in-Cell (CIC) Gewichte berechnen
        double wx[2] = {1.0 - dx, dx};
        double wy[2] = {1.0 - dy, dy};
        double wz[2] = {1.0 - dz, dz};

        // Kräfte von den umliegenden Gitterpunkten interpolieren
        vec3 force(0.0, 0.0, 0.0);

        for (int ii = 0; ii <= 1; ++ii)
        {
            int gi = safeMod(i + ii, gridSize);
            for (int jj = 0; jj <= 1; ++jj)
            {
                int gj = safeMod(j + jj, gridSize);
                for (int ll = 0; ll <= 1; ++ll) // Schleifenvariable umbenannt von k zu ll
                {
                    int gk = safeMod(k + ll, gridSize);
                    size_t index = static_cast<size_t>(gi * gridSize * gridSize + gj * gridSize + gk);

                    if (index >= forceGrid.size())
                    {
                        std::cerr << "Error: Index out of bounds in interpolateForcesToParticles()." << std::endl;
                        continue;
                    }

                    double weight = wx[ii] * wy[jj] * wz[ll];
                    force.x += forceGrid[index][0] * weight;
                    force.y += forceGrid[index][1] * weight;
                    force.z += forceGrid[index][2] * weight;
                }
            }
        }

        // Aktualisieren der Beschleunigung des Teilchens
        particle->acceleration = force;
    }
}
