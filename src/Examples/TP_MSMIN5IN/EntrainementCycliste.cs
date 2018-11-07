using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;


namespace TP_MSMIN5IN
{
	public class EntrainementCycliste : CyclisteBase
	{

		protected VariableArray<double> TempsDeTrajet;
		protected Variable<int> NombreDeTrajets;

		public override void CreationModeleBayesien()
		{
			base.CreationModeleBayesien();
			NombreDeTrajets = Variable.New<int>();
			Range indiceTrajet = new Range(NombreDeTrajets);
			TempsDeTrajet = Variable.Array<double>(indiceTrajet);
			using (Variable.ForEach(indiceTrajet))
			{
				TempsDeTrajet[indiceTrajet] = Variable.GaussianFromMeanAndPrecision(Moyenne, Bruit);
			}
		}

		public DonneesCycliste CalculePosterieurs(double[] donneesObservees)
		{
			DonneesCycliste posterieurs;
			NombreDeTrajets.ObservedValue = donneesObservees.Length;
			TempsDeTrajet.ObservedValue = donneesObservees;
			posterieurs.DistribMoyenne = MoteurInference.Infer<Gaussian>(Moyenne);
			posterieurs.DistribBruitTraffic = MoteurInference.Infer<Gamma>(Bruit);

			return posterieurs;
		}

	}

}
