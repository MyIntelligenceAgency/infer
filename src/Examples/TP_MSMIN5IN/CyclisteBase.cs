using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;


namespace TP_MSMIN5IN
{
    public class CyclisteBase
    {

        public InferenceEngine MoteurInference;

        protected Variable<double> Moyenne;
        protected Variable<double> Bruit;

        protected Variable<Gaussian> MoyenneAPriori;
        protected Variable<Gamma> BruitAPriori;

        public virtual void CreationModeleBayesien()
        {
            MoyenneAPriori = Variable.New<Gaussian>();
            BruitAPriori = Variable.New<Gamma>();
            Moyenne = Variable.Random<double, Gaussian>(MoyenneAPriori);
            Bruit = Variable.Random<double, Gamma>(BruitAPriori);
            if (MoteurInference == null)
            {
                MoteurInference = new InferenceEngine(new ExpectationPropagation());
            }
        }

        public virtual void DefinirDistributions(DonneesCycliste distribsApriori)
        {
            MoyenneAPriori.ObservedValue = distribsApriori.DistribMoyenne;
            BruitAPriori.ObservedValue = distribsApriori.DistribBruitTraffic;
        }

    }


}
