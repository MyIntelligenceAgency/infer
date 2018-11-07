using Microsoft.ML.Probabilistic.Distributions;


namespace TP_MSMIN5IN
{
    public struct DonneesCyclisteMixte
    {
        public Gaussian[] DistribMoyenne;
        public Gamma[] DistribBruitTraffic;
        public Dirichlet DistribMixe;

    }

}
