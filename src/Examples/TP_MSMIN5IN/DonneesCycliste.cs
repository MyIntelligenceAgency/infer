using Microsoft.ML.Probabilistic.Distributions;


namespace TP_MSMIN5IN
{
    public struct DonneesCycliste
    {
        public Gaussian DistribMoyenne;
        public Gamma DistribBruitTraffic;

        public DonneesCycliste(Gaussian moyenne, Gamma precision)
        {
            DistribMoyenne = moyenne;
            DistribBruitTraffic = precision;
        }
    }

}
