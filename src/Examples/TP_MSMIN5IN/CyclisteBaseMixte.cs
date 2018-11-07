using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;


namespace TP_MSMIN5IN
{
	public class CyclisteBaseMixte
	{

		public InferenceEngine MoteurInference;

		protected int NombreComposantes = 2;

		protected VariableArray<Gaussian> MoyennesAPriori;
		protected VariableArray<Gamma> BruitsAPriori;
		protected Variable<Dirichlet> MixeAPriori;

		protected VariableArray<double> Moyennes;
		protected VariableArray<double> Bruits;
		protected Variable<Vector> Mixe;


		public virtual void CreationModeleBayesien()
		{
			Range indiceComposants = new Range(NombreComposantes);
			MoteurInference = new InferenceEngine(new VariationalMessagePassing());

			MoyennesAPriori = Variable.Array<Gaussian>(indiceComposants);
			BruitsAPriori = Variable.Array<Gamma>(indiceComposants);
			Moyennes = Variable.Array<double>(indiceComposants);
			Bruits = Variable.Array<double>(indiceComposants);

			using (Variable.ForEach(indiceComposants))
			{
				Moyennes[indiceComposants] = Variable<double>.Random(MoyennesAPriori[indiceComposants]);
				Bruits[indiceComposants] = Variable<double>.Random(BruitsAPriori[indiceComposants]);
			}

			MixeAPriori = Variable.New<Dirichlet>();
			Mixe = Variable<Vector>.Random(MixeAPriori);
			Mixe.SetValueRange(indiceComposants);

		}

		public virtual void DefinirDistributions(
			DonneesCyclisteMixte distribsApriori)
		{
			MoyennesAPriori.ObservedValue = distribsApriori.DistribMoyenne;
			BruitsAPriori.ObservedValue = distribsApriori.DistribBruitTraffic;
			MixeAPriori.ObservedValue = distribsApriori.DistribMixe;
		}

	}


}
