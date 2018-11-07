using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;


namespace TP_MSMIN5IN
{
	public class EntrainementCyclisteMixte : CyclisteBaseMixte
	{

		protected Variable<int> NombreDeTrajets;

		protected VariableArray<double> TempsDeTrajet;
		protected VariableArray<int> ComposantesTrajets;


		public override void CreationModeleBayesien()
		{
			base.CreationModeleBayesien();
			NombreDeTrajets = Variable.New<int>();
			Range indiceTrajet = new Range(NombreDeTrajets);
			TempsDeTrajet = Variable.Array<double>(indiceTrajet);
			ComposantesTrajets = Variable.Array<int>(indiceTrajet);

			using (Variable.ForEach(indiceTrajet))
			{
				ComposantesTrajets[indiceTrajet] = Variable.Discrete(Mixe);
				using (Variable.Switch(ComposantesTrajets[indiceTrajet]))
				{
					TempsDeTrajet[indiceTrajet].SetTo(
					 Variable.GaussianFromMeanAndPrecision(
					  Moyennes[ComposantesTrajets[indiceTrajet]],
					  Bruits[ComposantesTrajets[indiceTrajet]]));
				}
			}
		}

		public DonneesCyclisteMixte CalculePosterieurs(
		   double[] donneesObservees)
		{
			DonneesCyclisteMixte posterieurs;
			NombreDeTrajets.ObservedValue = donneesObservees.Length;
			TempsDeTrajet.ObservedValue = donneesObservees;
			posterieurs.DistribMoyenne = MoteurInference.Infer<Gaussian[]>(Moyennes);
			posterieurs.DistribBruitTraffic = MoteurInference.Infer<Gamma[]>(Bruits);
			posterieurs.DistribMixe = MoteurInference.Infer<Dirichlet>(Mixe);

			return posterieurs;
		}
	}
}
