using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;


namespace TP_MSMIN5IN
{
	public class PredictionCyclisteMixte : CyclisteBaseMixte
	{
		private Gaussian demainDistrib;
		public Variable<double> demainTemps;

		public override void CreationModeleBayesien()
		{
			base.CreationModeleBayesien();
			Variable<int> indiceComposant = Variable.Discrete(Mixe);
			demainTemps = Variable.New<double>();
			using (Variable.Switch(indiceComposant))
			{
				demainTemps.SetTo(Variable.GaussianFromMeanAndPrecision(Moyennes[indiceComposant], Bruits[indiceComposant]));
			}
		}

		public Gaussian EstimerTempsDemain()
		{
			demainDistrib = MoteurInference.Infer<Gaussian>(demainTemps);
			return demainDistrib;
		}

	}
}
