// <auto-generated />
#pragma warning disable 1570, 1591

using System;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal
{
	public partial class GaussianDenseBinaryBpmPrediction_EP : IGeneratedAlgorithm
	{
		#region Fields
		/// <summary>True if Changed_FeatureCount_FeatureValues_InstanceCount_WeightConstraints_WeightPriors has executed. Set this to false to force re-execution of Changed_FeatureCount_FeatureValues_InstanceCount_WeightConstraints_WeightPriors</summary>
		public bool Changed_FeatureCount_FeatureValues_InstanceCount_WeightConstraints_WeightPriors_isDone;
		/// <summary>True if Changed_FeatureCount_InstanceCount has executed. Set this to false to force re-execution of Changed_FeatureCount_InstanceCount</summary>
		public bool Changed_FeatureCount_InstanceCount_isDone;
		/// <summary>True if Changed_FeatureCount_InstanceCount_WeightConstraints_WeightPriors has executed. Set this to false to force re-execution of Changed_FeatureCount_InstanceCount_WeightConstraints_WeightPriors</summary>
		public bool Changed_FeatureCount_InstanceCount_WeightConstraints_WeightPriors_isDone;
		/// <summary>True if Changed_FeatureCount has executed. Set this to false to force re-execution of Changed_FeatureCount</summary>
		public bool Changed_FeatureCount_isDone;
		/// <summary>True if Changed_FeatureCount_WeightConstraints_WeightPriors has executed. Set this to false to force re-execution of Changed_FeatureCount_WeightConstraints_WeightPriors</summary>
		public bool Changed_FeatureCount_WeightConstraints_WeightPriors_isDone;
		/// <summary>True if Changed_FeatureCount_WeightPriors has executed. Set this to false to force re-execution of Changed_FeatureCount_WeightPriors</summary>
		public bool Changed_FeatureCount_WeightPriors_isDone;
		/// <summary>True if Changed_InstanceCount has executed. Set this to false to force re-execution of Changed_InstanceCount</summary>
		public bool Changed_InstanceCount_isDone;
		/// <summary>True if Changed_WeightConstraints_WeightPriors has executed. Set this to false to force re-execution of Changed_WeightConstraints_WeightPriors</summary>
		public bool Changed_WeightConstraints_WeightPriors_isDone;
		/// <summary>True if Changed_WeightPriors has executed. Set this to false to force re-execution of Changed_WeightPriors</summary>
		public bool Changed_WeightPriors_isDone;
		/// <summary>True if Constant has executed. Set this to false to force re-execution of Constant</summary>
		public bool Constant_isDone;
		/// <summary>Field backing the FeatureCount property</summary>
		private int featureCount;
		public DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]> FeatureScores_F;
		/// <summary>Field backing the FeatureValues property</summary>
		private double[][] featureValues;
		/// <summary>Field backing the InstanceCount property</summary>
		private int instanceCount;
		public DistributionStructArray<Bernoulli,bool> Labels_F;
		/// <summary>Message to marginal of 'Labels'</summary>
		public DistributionStructArray<Bernoulli,bool> Labels_marginal_F;
		public Bernoulli Labels_use_B_reduced;
		public DistributionStructArray<Gaussian,double> NoisyScore_F;
		/// <summary>Field backing the NumberOfIterationsDone property</summary>
		private int numberOfIterationsDone;
		public DistributionStructArray<Gaussian,double> Score_F;
		/// <summary>Field backing the WeightConstraints property</summary>
		private DistributionStructArray<Gaussian,double> weightConstraints;
		/// <summary>Field backing the WeightPriors property</summary>
		private DistributionStructArray<Gaussian,double> weightPriors;
		public DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]> Weights_depth1_rep_B;
		/// <summary>Buffer for ReplicateOp_Divide.Marginal&lt;Gaussian&gt;</summary>
		public DistributionStructArray<Gaussian,double> Weights_depth1_rep_B_toDef;
		public DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]> Weights_depth1_rep_F;
		/// <summary>Buffer for ReplicateOp_Divide.UsesAverageConditional&lt;Gaussian&gt;</summary>
		public DistributionStructArray<Gaussian,double> Weights_depth1_rep_F_marginal;
		/// <summary>Messages from use of 'Weights'</summary>
		public DistributionStructArray<Gaussian,double>[] Weights_uses_B;
		/// <summary>Messages to use of 'Weights'</summary>
		public DistributionStructArray<Gaussian,double>[] Weights_uses_F;
		#endregion

		#region Properties
		/// <summary>The externally-specified value of 'FeatureCount'</summary>
		public int FeatureCount
		{
			get {
				return this.featureCount;
			}
			set {
				if (this.featureCount!=value) {
					this.featureCount = value;
					this.numberOfIterationsDone = 0;
					this.Changed_FeatureCount_isDone = false;
					this.Changed_FeatureCount_InstanceCount_isDone = false;
					this.Changed_FeatureCount_WeightPriors_isDone = false;
					this.Changed_FeatureCount_WeightConstraints_WeightPriors_isDone = false;
					this.Changed_FeatureCount_InstanceCount_WeightConstraints_WeightPriors_isDone = false;
					this.Changed_FeatureCount_FeatureValues_InstanceCount_WeightConstraints_WeightPriors_isDone = false;
				}
			}
		}

		/// <summary>The externally-specified value of 'FeatureValues'</summary>
		public double[][] FeatureValues
		{
			get {
				return this.featureValues;
			}
			set {
				if ((value!=null)&&(value.Length!=this.instanceCount)) {
					throw new ArgumentException(((("Provided array of length "+value.Length)+" when length ")+this.instanceCount)+" was expected for variable \'FeatureValues\'");
				}
				this.featureValues = value;
				this.numberOfIterationsDone = 0;
				this.Changed_FeatureCount_FeatureValues_InstanceCount_WeightConstraints_WeightPriors_isDone = false;
			}
		}

		/// <summary>The externally-specified value of 'InstanceCount'</summary>
		public int InstanceCount
		{
			get {
				return this.instanceCount;
			}
			set {
				if (this.instanceCount!=value) {
					this.instanceCount = value;
					this.numberOfIterationsDone = 0;
					this.Changed_InstanceCount_isDone = false;
					this.Changed_FeatureCount_InstanceCount_isDone = false;
					this.Changed_FeatureCount_InstanceCount_WeightConstraints_WeightPriors_isDone = false;
					this.Changed_FeatureCount_FeatureValues_InstanceCount_WeightConstraints_WeightPriors_isDone = false;
				}
			}
		}

		/// <summary>The number of iterations done from the initial state</summary>
		public int NumberOfIterationsDone
		{
			get {
				return this.numberOfIterationsDone;
			}
		}

		/// <summary>The externally-specified value of 'WeightConstraints'</summary>
		public DistributionStructArray<Gaussian,double> WeightConstraints
		{
			get {
				return this.weightConstraints;
			}
			set {
				this.weightConstraints = value;
				this.numberOfIterationsDone = 0;
				this.Changed_WeightConstraints_WeightPriors_isDone = false;
				this.Changed_FeatureCount_WeightConstraints_WeightPriors_isDone = false;
				this.Changed_FeatureCount_InstanceCount_WeightConstraints_WeightPriors_isDone = false;
				this.Changed_FeatureCount_FeatureValues_InstanceCount_WeightConstraints_WeightPriors_isDone = false;
			}
		}

		/// <summary>The externally-specified value of 'WeightPriors'</summary>
		public DistributionStructArray<Gaussian,double> WeightPriors
		{
			get {
				return this.weightPriors;
			}
			set {
				this.weightPriors = value;
				this.numberOfIterationsDone = 0;
				this.Changed_WeightPriors_isDone = false;
				this.Changed_FeatureCount_WeightPriors_isDone = false;
				this.Changed_WeightConstraints_WeightPriors_isDone = false;
				this.Changed_FeatureCount_WeightConstraints_WeightPriors_isDone = false;
				this.Changed_FeatureCount_InstanceCount_WeightConstraints_WeightPriors_isDone = false;
				this.Changed_FeatureCount_FeatureValues_InstanceCount_WeightConstraints_WeightPriors_isDone = false;
			}
		}

		#endregion

		#region Methods
		/// <summary>Computations that depend on the observed value of FeatureCount</summary>
		private void Changed_FeatureCount()
		{
			if (this.Changed_FeatureCount_isDone) {
				return ;
			}
			this.Weights_depth1_rep_F_marginal = new DistributionStructArray<Gaussian,double>(this.featureCount);
			this.Weights_depth1_rep_B_toDef = new DistributionStructArray<Gaussian,double>(this.featureCount);
			this.Weights_depth1_rep_F = new DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]>(this.featureCount);
			this.Weights_depth1_rep_B = new DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]>(this.featureCount);
			this.Changed_FeatureCount_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of FeatureCount and FeatureValues and InstanceCount and WeightConstraints and WeightPriors</summary>
		private void Changed_FeatureCount_FeatureValues_InstanceCount_WeightConstraints_WeightPriors()
		{
			if (this.Changed_FeatureCount_FeatureValues_InstanceCount_WeightConstraints_WeightPriors_isDone) {
				return ;
			}
			for(int InstanceRange = 0; InstanceRange<this.instanceCount; InstanceRange++) {
				for(int FeatureRange = 0; FeatureRange<this.featureCount; FeatureRange++) {
					this.FeatureScores_F[InstanceRange][FeatureRange] = GaussianProductOpBase.ProductAverageConditional(this.featureValues[InstanceRange][FeatureRange], this.Weights_depth1_rep_F[FeatureRange][InstanceRange]);
				}
				this.Score_F[InstanceRange] = FastSumOp.SumAverageConditional(this.FeatureScores_F[InstanceRange]);
				this.NoisyScore_F[InstanceRange] = GaussianFromMeanAndVarianceOp.SampleAverageConditional(this.Score_F[InstanceRange], 1.0);
				this.Labels_F[InstanceRange] = IsPositiveOp.IsPositiveAverageConditional(this.NoisyScore_F[InstanceRange]);
				this.Labels_marginal_F[InstanceRange] = DerivedVariableOp.MarginalAverageConditional<Bernoulli>(this.Labels_use_B_reduced, this.Labels_F[InstanceRange], this.Labels_marginal_F[InstanceRange]);
			}
			this.Changed_FeatureCount_FeatureValues_InstanceCount_WeightConstraints_WeightPriors_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of FeatureCount and InstanceCount</summary>
		private void Changed_FeatureCount_InstanceCount()
		{
			if (this.Changed_FeatureCount_InstanceCount_isDone) {
				return ;
			}
			for(int FeatureRange = 0; FeatureRange<this.featureCount; FeatureRange++) {
				this.Weights_depth1_rep_F[FeatureRange] = new DistributionStructArray<Gaussian,double>(this.instanceCount);
				this.Weights_depth1_rep_B[FeatureRange] = new DistributionStructArray<Gaussian,double>(this.instanceCount);
			}
			for(int InstanceRange = 0; InstanceRange<this.instanceCount; InstanceRange++) {
				for(int FeatureRange = 0; FeatureRange<this.featureCount; FeatureRange++) {
					this.Weights_depth1_rep_B[FeatureRange][InstanceRange] = Gaussian.Uniform();
					this.Weights_depth1_rep_F[FeatureRange][InstanceRange] = Gaussian.Uniform();
				}
				this.FeatureScores_F[InstanceRange] = new DistributionStructArray<Gaussian,double>(this.featureCount);
				for(int FeatureRange = 0; FeatureRange<this.featureCount; FeatureRange++) {
					this.FeatureScores_F[InstanceRange][FeatureRange] = Gaussian.Uniform();
				}
			}
			this.Changed_FeatureCount_InstanceCount_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of FeatureCount and InstanceCount and WeightConstraints and WeightPriors</summary>
		private void Changed_FeatureCount_InstanceCount_WeightConstraints_WeightPriors()
		{
			if (this.Changed_FeatureCount_InstanceCount_WeightConstraints_WeightPriors_isDone) {
				return ;
			}
			for(int InstanceRange = 0; InstanceRange<this.instanceCount; InstanceRange++) {
				for(int FeatureRange = 0; FeatureRange<this.featureCount; FeatureRange++) {
					this.Weights_depth1_rep_F[FeatureRange][InstanceRange] = ReplicateOp_Divide.UsesAverageConditional<Gaussian>(this.Weights_depth1_rep_B[FeatureRange][InstanceRange], this.Weights_depth1_rep_F_marginal[FeatureRange], InstanceRange, this.Weights_depth1_rep_F[FeatureRange][InstanceRange]);
				}
			}
			this.Changed_FeatureCount_InstanceCount_WeightConstraints_WeightPriors_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of FeatureCount and WeightConstraints and WeightPriors</summary>
		private void Changed_FeatureCount_WeightConstraints_WeightPriors()
		{
			if (this.Changed_FeatureCount_WeightConstraints_WeightPriors_isDone) {
				return ;
			}
			for(int FeatureRange = 0; FeatureRange<this.featureCount; FeatureRange++) {
				this.Weights_depth1_rep_F_marginal[FeatureRange] = ReplicateOp_Divide.Marginal<Gaussian>(this.Weights_depth1_rep_B_toDef[FeatureRange], this.Weights_uses_F[1][FeatureRange], this.Weights_depth1_rep_F_marginal[FeatureRange]);
			}
			this.Changed_FeatureCount_WeightConstraints_WeightPriors_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of FeatureCount and WeightPriors</summary>
		private void Changed_FeatureCount_WeightPriors()
		{
			if (this.Changed_FeatureCount_WeightPriors_isDone) {
				return ;
			}
			for(int FeatureRange = 0; FeatureRange<this.featureCount; FeatureRange++) {
				this.Weights_depth1_rep_F_marginal[FeatureRange] = ReplicateOp_Divide.MarginalInit<Gaussian>(this.Weights_uses_F[1][FeatureRange]);
				this.Weights_depth1_rep_B_toDef[FeatureRange] = ReplicateOp_Divide.ToDefInit<Gaussian>(this.Weights_uses_F[1][FeatureRange]);
			}
			this.Changed_FeatureCount_WeightPriors_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of InstanceCount</summary>
		private void Changed_InstanceCount()
		{
			if (this.Changed_InstanceCount_isDone) {
				return ;
			}
			this.Labels_F = new DistributionStructArray<Bernoulli,bool>(this.instanceCount);
			this.NoisyScore_F = new DistributionStructArray<Gaussian,double>(this.instanceCount);
			this.Score_F = new DistributionStructArray<Gaussian,double>(this.instanceCount);
			this.FeatureScores_F = new DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]>(this.instanceCount);
			this.Labels_marginal_F = new DistributionStructArray<Bernoulli,bool>(this.instanceCount);
			this.Labels_use_B_reduced = default(Bernoulli);
			if (this.instanceCount>0) {
				this.Labels_use_B_reduced = Bernoulli.Uniform();
			}
			for(int InstanceRange = 0; InstanceRange<this.instanceCount; InstanceRange++) {
				this.Labels_F[InstanceRange] = Bernoulli.Uniform();
				this.Score_F[InstanceRange] = Gaussian.Uniform();
				this.NoisyScore_F[InstanceRange] = Gaussian.Uniform();
				this.Labels_marginal_F[InstanceRange] = Bernoulli.Uniform();
			}
			this.Changed_InstanceCount_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of WeightConstraints and WeightPriors</summary>
		private void Changed_WeightConstraints_WeightPriors()
		{
			if (this.Changed_WeightConstraints_WeightPriors_isDone) {
				return ;
			}
			this.Weights_uses_B[0] = ArrayHelper.SetTo<DistributionStructArray<Gaussian,double>>(this.Weights_uses_B[0], this.weightConstraints);
			this.Weights_uses_F[1] = ReplicateOp_NoDivide.UsesAverageConditional<DistributionStructArray<Gaussian,double>>(this.Weights_uses_B, this.weightPriors, 1, this.Weights_uses_F[1]);
			this.Changed_WeightConstraints_WeightPriors_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of WeightPriors</summary>
		private void Changed_WeightPriors()
		{
			if (this.Changed_WeightPriors_isDone) {
				return ;
			}
			this.Weights_uses_B[0] = ArrayHelper.MakeUniform<DistributionStructArray<Gaussian,double>>(this.weightPriors);
			this.Weights_uses_F[1] = ArrayHelper.MakeUniform<DistributionStructArray<Gaussian,double>>(this.weightPriors);
			this.Changed_WeightPriors_isDone = true;
		}

		/// <summary>Computations that do not depend on observed values</summary>
		private void Constant()
		{
			if (this.Constant_isDone) {
				return ;
			}
			this.Weights_uses_F = new DistributionStructArray<Gaussian,double>[2];
			this.Weights_uses_B = new DistributionStructArray<Gaussian,double>[2];
			this.Constant_isDone = true;
		}

		/// <summary>Update all marginals, by iterating message passing the given number of times</summary>
		/// <param name="numberOfIterations">The number of times to iterate each loop</param>
		/// <param name="initialise">If true, messages that initialise loops are reset when observed values change</param>
		private void Execute(int numberOfIterations, bool initialise)
		{
			this.Changed_FeatureCount();
			this.Constant();
			this.Changed_InstanceCount();
			this.Changed_FeatureCount_InstanceCount();
			this.Changed_WeightPriors();
			this.Changed_FeatureCount_WeightPriors();
			this.Changed_WeightConstraints_WeightPriors();
			this.Changed_FeatureCount_WeightConstraints_WeightPriors();
			this.Changed_FeatureCount_InstanceCount_WeightConstraints_WeightPriors();
			this.Changed_FeatureCount_FeatureValues_InstanceCount_WeightConstraints_WeightPriors();
			this.numberOfIterationsDone = numberOfIterations;
		}

		/// <summary>Update all marginals, by iterating message-passing the given number of times</summary>
		/// <param name="numberOfIterations">The total number of iterations that should be executed for the current set of observed values.  If this is more than the number already done, only the extra iterations are done.  If this is less than the number already done, message-passing is restarted from the beginning.  Changing the observed values resets the iteration count to 0.</param>
		public void Execute(int numberOfIterations)
		{
			this.Execute(numberOfIterations, true);
		}

		/// <summary>Get the observed value of the specified variable.</summary>
		/// <param name="variableName">Variable name</param>
		public object GetObservedValue(string variableName)
		{
			if (variableName=="InstanceCount") {
				return this.InstanceCount;
			}
			if (variableName=="FeatureCount") {
				return this.FeatureCount;
			}
			if (variableName=="FeatureValues") {
				return this.FeatureValues;
			}
			if (variableName=="WeightPriors") {
				return this.WeightPriors;
			}
			if (variableName=="WeightConstraints") {
				return this.WeightConstraints;
			}
			throw new ArgumentException("Not an observed variable name: "+variableName);
		}

		/// <summary>
		/// Returns the marginal distribution for 'Labels' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public DistributionStructArray<Bernoulli,bool> LabelsMarginal()
		{
			return this.Labels_marginal_F;
		}

		/// <summary>Get the marginal distribution (computed up to this point) of a variable</summary>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <returns>The marginal distribution computed up to this point</returns>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public object Marginal(string variableName)
		{
			if (variableName=="Labels") {
				return this.LabelsMarginal();
			}
			throw new ArgumentException("This class was not built to infer "+variableName);
		}

		/// <summary>Get the marginal distribution (computed up to this point) of a variable, converted to type T</summary>
		/// <typeparam name="T">The distribution type.</typeparam>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <returns>The marginal distribution computed up to this point</returns>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public T Marginal<T>(string variableName)
		{
			return Distribution.ChangeType<T>(this.Marginal(variableName));
		}

		/// <summary>Get the query-specific marginal distribution of a variable.</summary>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <param name="query">QueryType name. For example, GibbsSampling answers 'Marginal', 'Samples', and 'Conditionals' queries</param>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public object Marginal(string variableName, string query)
		{
			if (query=="Marginal") {
				return this.Marginal(variableName);
			}
			throw new ArgumentException(((("This class was not built to infer \'"+variableName)+"\' with query \'")+query)+"\'");
		}

		/// <summary>Get the query-specific marginal distribution of a variable, converted to type T</summary>
		/// <typeparam name="T">The distribution type.</typeparam>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <param name="query">QueryType name. For example, GibbsSampling answers 'Marginal', 'Samples', and 'Conditionals' queries</param>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public T Marginal<T>(string variableName, string query)
		{
			return Distribution.ChangeType<T>(this.Marginal(variableName, query));
		}

		private void OnProgressChanged(ProgressChangedEventArgs e)
		{
			// Make a temporary copy of the event to avoid a race condition
			// if the last subscriber unsubscribes immediately after the null check and before the event is raised.
			EventHandler<ProgressChangedEventArgs> handler = this.ProgressChanged;
			if (handler!=null) {
				handler(this, e);
			}
		}

		/// <summary>Reset all messages to their initial values.  Sets NumberOfIterationsDone to 0.</summary>
		public void Reset()
		{
			this.Execute(0);
		}

		/// <summary>Set the observed value of the specified variable.</summary>
		/// <param name="variableName">Variable name</param>
		/// <param name="value">Observed value</param>
		public void SetObservedValue(string variableName, object value)
		{
			if (variableName=="InstanceCount") {
				this.InstanceCount = (int)value;
				return ;
			}
			if (variableName=="FeatureCount") {
				this.FeatureCount = (int)value;
				return ;
			}
			if (variableName=="FeatureValues") {
				this.FeatureValues = (double[][])value;
				return ;
			}
			if (variableName=="WeightPriors") {
				this.WeightPriors = (DistributionStructArray<Gaussian,double>)value;
				return ;
			}
			if (variableName=="WeightConstraints") {
				this.WeightConstraints = (DistributionStructArray<Gaussian,double>)value;
				return ;
			}
			throw new ArgumentException("Not an observed variable name: "+variableName);
		}

		/// <summary>Update all marginals, by iterating message-passing an additional number of times</summary>
		/// <param name="additionalIterations">The number of iterations that should be executed, starting from the current message state.  Messages are not reset, even if observed values have changed.</param>
		public void Update(int additionalIterations)
		{
			this.Execute(checked(this.numberOfIterationsDone+additionalIterations), false);
		}

		#endregion

		#region Events
		/// <summary>Event that is fired when the progress of inference changes, typically at the end of one iteration of the inference algorithm.</summary>
		public event EventHandler<ProgressChangedEventArgs> ProgressChanged;
		#endregion

	}

}
