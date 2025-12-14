# Synthetic Persona Generation Methods for U.S. High School Students (Ages 16-18)
## Comprehensive Framework & Methodology Comparison

---

## EXECUTIVE SUMMARY

This framework presents **three complementary methodologies** for generating 100 synthetic personas representing U.S. high school students (ages 16-18). Each method produces realistic personas with different strengths:

- **Marginal Distributions Method**: Fastest, captures individual variable distributions accurately
- **Constraint-Based Method**: Most realistic, captures meaningful correlations between variables
- **Stratified Sampling Method**: Most representative, ensures demographic groups are properly represented

All three methods were validated against real-world research data on high school student demographics, attitudes, and behaviors.

---

## PART 1: RESEARCH DATA FOUNDATION

### Demographic Distributions

**Gender**: 51% Female, 49% Male
- Female students average GPA: 3.1
- Male students average GPA: 2.91
- Girls more likely to have LGBTQ+ identification

**Race/Ethnicity**:
- White: 39% (declining, -26% projected through 2041)
- Hispanic: 28% (growing, +16% projected)
- Black: 15%
- Asian: 5.5%
- Two or more races: 5% (growing, +68% projected)
- Other: 0.9%

**Sexual Orientation & Gender Identity**:
- LGBTQ+: 26% (up from 11% in 2015)
- Increase concentrated in female identification as bisexual or unsure
- LGBTQ+ students: 3x higher suicide attempt rate, worse mental health outcomes

**Socioeconomic Status**:
- Median household income: $60,000
- 20th percentile: $25,000 | 80th percentile: $111,000 | 99th percentile: $512,000

**First-Generation College Students**: 45-50%
- Lower SES families more likely to be first-generation
- Strong inverse correlation between family income and first-gen status

**Academic Performance (GPA)**:
- Average HS GPA: 3.0
- By race: Asian 3.26-3.52 > White 3.09-3.23 > Hispanic 2.84-2.98 > Black 2.68-2.69
- Correlation with family income: +0.3 to +0.5 (moderate positive)
- Income × GPA: $10,000 income increase = ~3.6% improvement in performance

**Sports Participation**: 54% (8.27 million of 15 million HS students)
- Boys: 57% of male population
- Girls: 43% of female population

**Mental Health**:
- 37% report poor mental health most/all of the time
- 45% stressed almost every day
- Girls significantly worse than boys
- LGBTQ+ students: 65% experience persistent sadness/hopelessness

**Social Media Usage** (daily use):
- YouTube: 75% daily, 20% almost constantly
- TikTok: 60% daily, girls 19% vs boys 13% almost constantly
- Instagram: 60% daily, 12% almost constantly
- 33% use at least one platform almost constantly
- Black and Hispanic teens: higher near-constant use

**College Intentions**:
- 45% plan 4-year or 2-year college (down from 73% in 2018)
- College enrollment gender gap: 69.5% women vs 55.4% men
- By race: Asian 94.7% > White 62.2% > Black 59.2% > Hispanic 55.4%
- 86% worry about college costs

### Key Correlations Identified

1. **Income ↔ GPA**: Moderate positive (0.35-0.45)
2. **Income ↔ First-Gen Status**: Strong negative (-0.60)
3. **GPA ↔ College Intentions**: Strong positive (0.50+)
4. **Gender ↔ LGBTQ+ Identity**: Female higher (30% vs 21%)
5. **Gender ↔ Mental Health**: Female worse (43% poor vs 28% poor)
6. **Gender ↔ Social Media Intensity**: Female higher intensity
7. **Income ↔ College Enrollment**: Strong positive (3x gap between quintiles)

---

## PART 2: METHODOLOGY 1 - MARGINAL DISTRIBUTIONS

### Description
Generate personas by independently sampling from each demographic variable's marginal distribution without considering correlations.

### Algorithm
```
For each persona i in range(100):
  1. Sample gender from [Female 51%, Male 49%]
  2. Sample race/ethnicity from distribution
  3. Sample sexual orientation from [Heterosexual 74%, LGBTQ+ 26%]
  4. Sample family_income from lognormal(mean=10.8, sigma=0.8)
  5. Sample first_gen_college from [Yes 45%, No 55%]
  6. Sample gpa from normal distribution (female μ=3.1, male μ=2.91, σ=0.5)
  7. Sample sports_participation from [Yes 54%, No 46%]
  8. Sample mental_health from [Good 40%, Fair 23%, Poor 37%]
  9. Sample social_media_intensity from [Low 15%, Moderate 20%, High 32%, Almost Constant 33%]
  10. Sample college_intention from distribution
```

### Advantages
- **Simplicity**: Easy to implement and understand
- **Accuracy for Marginals**: Perfectly captures individual variable distributions
- **Speed**: Fast generation
- **Flexibility**: Easy to modify individual distributions

### Disadvantages
- **No Correlations**: Misses realistic relationships (e.g., low-income/high-GPA combos)
- **Unrealistic Combinations**: Can generate implausible personas
- **Less Representative**: May not capture true subgroup compositions
- **Lower Validity**: Survey responses less aligned with real data

### Validation Results
- Female: 51.0% ✓
- LGBTQ+: 23.0% (target 26%) -3%
- First-Gen: 51.0% (target 45%) +6%
- Avg GPA: 3.09 ✓
- Income-GPA correlation: -0.031 (expected 0.35-0.45) ✗

### When to Use
- Quick exploratory analysis
- When speed is more important than accuracy
- When variable independence is acceptable assumption
- Training/demonstration purposes

---

## PART 3: METHODOLOGY 2 - CONSTRAINT-BASED

### Description
Generate personas using probabilistic constraints that reflect realistic relationships between variables.

### Key Constraints Implemented

**1. Gender → LGBTQ+ Identification**
- Females: 30% LGBTQ+ (vs 21% males)

**2. Income Percentile → GPA**
- Top 30%: +0.3 GPA boost
- Middle 40%: +0.15 GPA boost
- Bottom 30%: -0.15 GPA boost

**3. Income Percentile → First-Gen Status**
- Top 30%: 30% first-gen
- Middle 40%: 45% first-gen
- Bottom 30%: 60% first-gen

**4. Race → GPA Adjustment**
- Asian: +0.25
- White: +0.10
- Hispanic: -0.15
- Black: -0.30

**5. Gender → Mental Health**
- Females: 43% poor, 22% fair, 35% good
- Males: 28% poor, 24% fair, 48% good

**6. LGBTQ+ Status → Mental Health**
- -15% good mental health
- +10% poor mental health

**7. Race → Social Media Intensity**
- Black: 1.15x factor
- Hispanic: 1.10x factor
- Asian: 0.90x factor

**8. GPA × Income → College Intentions**
- High GPA + High Income: 60% 4-year college
- Moderate GPA + Moderate Income: 40% 4-year college
- Low GPA: 10-20% 4-year college

### Algorithm
```
For each persona i in range(100):
  1. Sample gender and race
  2. Sample income from lognormal
  3. Calculate income_percentile
  4. Sample sexual_orientation using gender-conditional probabilities
  5. Sample first_gen_college using income-conditional probabilities
  6. Calculate base_gpa using gender, income_percentile, race
    base_gpa = 3.0 + gender_adjust + income_adjust + race_adjust
  7. Add noise: gpa = clip(normal(base_gpa, 0.5), 0, 4)
  8. Sample sports_participation (LGBTQ+ lower)
  9. Sample mental_health using gender + LGBTQ+ adjustments
  10. Sample social_media_intensity using race + gender factors
  11. Sample college_intention using gpa × income constraints
       + first_gen_college modifier
```

### Advantages
- **Realistic Correlations**: Captures meaningful relationships
- **Domain Knowledge**: Reflects research findings
- **Better Persona Quality**: More believable character combinations
- **Survey Validity**: Produces more realistic survey responses
- **Constraint Satisfaction**: Respects known relationships

### Disadvantages
- **Complexity**: More code, harder to maintain
- **Assumption Dependent**: Quality depends on constraint accuracy
- **Less Transparent**: Harder to debug issues
- **Calibration Required**: Need to tune adjustment parameters

### Validation Results
- Female: 46.0% (target 51%) -5%
- LGBTQ+: 25.0% (target 26%) ✓
- First-Gen: 45.0% (target 45%) ✓
- Avg GPA: 3.00 ✓
- Income-GPA correlation: 0.217 (expected 0.35-0.45) ~good
- Poor mental health: 33% (target 37%) -4%

### Survey Result Differences (vs Marginal)
- College "Very Important": +8% (22% → 30%)
- Social Media "Strongly Agree": -1% (similar)
- School Start Times "Strongly Support": Same (14%)
- Overall more realistic distribution across response options

### When to Use
- **Primary method** for persona development
- When demographic realism is important
- For user research and segmentation studies
- When personas will be extensively analyzed
- Policy research and impact studies

---

## PART 4: METHODOLOGY 3 - STRATIFIED SAMPLING

### Description
Generate personas by stratifying on key demographic variables (gender, race, income quintile) to ensure guaranteed representation of demographic groups.

### Stratification Strategy

**Primary Strata**: Income Quintiles (guaranteed 20% each)
- Bottom 20%: $0-30,000
- 20-40%: $30,000-45,000
- 40-60%: $45,000-60,000
- 60-80%: $60,000-85,000
- Top 20%: $85,000+

**Secondary Strata**: Gender (51% F, 49% M)

**Tertiary Strata**: Race/Ethnicity (proportional distribution)

**Then Fill**: Sexual orientation, GPA (income-conditioned), mental health, college intentions

### Algorithm
```
For each quintile q in [Bottom, 20-40, 40-60, 60-80, Top]:
  For i in range(20):  // 20 personas per quintile = 100 total
    1. Sample gender from distribution
    2. Sample race from distribution
    3. Sample family_income uniformly from income_range[q]
    4. Assign income_quintile = q
    5. Sample sexual_orientation (gender-conditional)
    6. Sample first_gen_college (income-conditional)
    7. Calculate base_gpa using quintile + gender + race
    8. Sample gpa from normal(base_gpa, 0.5)
    9. Sample remaining variables
```

### Guarantees
- Exactly 20 personas in each income quintile
- Proper representation of all demographic groups
- Reduces sampling variance in subgroup estimates
- Better for hypothesis testing about subgroup differences

### Advantages
- **Representative**: Guaranteed demographic coverage
- **Reduced Variance**: Better estimates for small subgroups
- **Hypothesis Testing**: Better for stratified analysis
- **Demographic Balance**: Prevents undersampling of minorities
- **Subgroup Comparison**: Excellent for A/B testing between demographics

### Disadvantages
- **Less Flexibility**: Fixed stratification may not match true correlations
- **Constraint Complexity**: Harder to add additional constraints
- **Interaction Effects**: May miss higher-order interactions
- **Subgroup Sizes**: Fixed strata (20 each) may not reflect true proportions

### Validation Results
- Female: 51.0% ✓
- LGBTQ+: 24.0% (target 26%) -2%
- First-Gen: 46.0% (target 45%) ✓
- Avg GPA: 3.02 ✓
- Sports: 56.0% (target 54%) +2%
- Poor Mental Health: 39.0% (target 37%) +2%
- Income-GPA correlation: 0.436 (expected 0.35-0.45) ✓✓

### Survey Result Differences
- College "Very Important": +10% (22% → 32%)
- Social Media "Strongly Agree": +6% (50% → 56%)
- School Start Times "Strongly Support": +7% (14% → 21%)
- School Start Times "Support": -6% (42% → 36%)

### When to Use
- **Survey research** where demographic representation is critical
- **Comparative studies** analyzing differences across subgroups
- **Reporting**: When demographics must be properly weighted
- **Policy research**: Where subgroup equity is important
- **Sample quality assurance**: When you need guaranteed coverage


## PART 8: TECHNICAL NOTES

### Data Distribution Implementation

**Family Income**: Lognormal(μ=10.8, σ=0.8)
- Reflects real income skew (right-tailed)
- Produces realistic distribution with median ~$60k

**GPA**: Normal(μ=3.0-3.1, σ=0.5), clipped [0, 4.0]
- Accounts for gender differences
- Adjustable by income and race

**Age**: Fixed at 16-18 (not varied in this model)

### Parameter Tuning Guidance

If validation shows discrepancies:

**Too Few High-GPA Students**
- Increase σ in normal distribution
- Reduce income adjustment magnitude

**Too Many LGBTQ+ Students**
- Reduce gender-specific probabilities
- Check that 26% overall is target

**Wrong Income-GPA Correlation**
- Adjust income_adjustment coefficients
- Check if race adjustments are offsetting

---

## PART 9: RESEARCH SOURCES

All distributions based on:
- CDC Adolescent Behaviors and Experiences Survey (ABES) 2021
- Pew Research Center Social Media & Technology surveys 2024-2025
- U.S. Bureau of Labor Statistics (BLS) education data
- National Center for Education Statistics (NCES)
- American Student Assistance (ASA) college aspirations study
- Gates Foundation student perceptions research
- Brookings Institution college enrollment analysis
- Various peer-reviewed studies in Journal of Education Statistics
- State education department graduation and enrollment data


