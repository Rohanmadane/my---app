import streamlit as st
import requests
import py3Dmol
import stmol
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem import rdMolDescriptors
import pandas as pd
import joblib
from admet_ai import ADMETModel




# =========================================================
# GET SMILES FROM PUBCHEM
# =========================================================

def get_smiles(drug_name):

    try:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{drug_name}/property/ConnectivitySMILES/JSON"
        )

        response = requests.get(url, timeout=10)

        if response.status_code == 200:

            data = response.json()

            return data["PropertyTable"]["Properties"][0][
                "ConnectivitySMILES"
            ]

    except Exception:
        pass

    return None




# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Drug Discovery Portal",
    page_icon="🧪",
    layout="wide"
)


# =========================================================
# LOAD ADMET MODEL
# =========================================================

@st.cache_resource
def load_admet_model():
    return ADMETModel()


try:

    admet_model = load_admet_model()
    admet_loaded = True

except Exception as e:

    admet_loaded = False
    admet_model = None
    st.error(f"ADMET model loading error: {e}")


# =========================================================
# LOAD ML MODEL
# =========================================================

try:

    model_path = "drug_likeness_model.pkl"

    model = joblib.load(model_path)

    model_loaded = True

except Exception as e:

    model = None
    model_loaded = False

    st.error(f"ML model loading error: {e}")


# =========================================================
# TITLE
# =========================================================

st.title("🧪 Computational Analysis of Molecular Properties and Drug-Likeness Assessment")

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("🔬 Input Chemical Structure")

drug_name = st.sidebar.text_input(
    "Enter Drug Name:",
    ""
)

structure_type = st.sidebar.radio(
    "Select Structure:",
    ["2D", "3D"],
    horizontal=True
)

smiles_input = ""

if drug_name:

    smiles_input = get_smiles(drug_name)
    
    

    if smiles_input:

        st.sidebar.text_input(
            "SMILES Code:",
            value=smiles_input,
            disabled=True
        )
        

    else:

        st.sidebar.error("Drug not found.")

else:

    st.sidebar.warning(
        "Enter a drug name to fetch its SMILES."
    )

st.sidebar.caption(
    "Enter a valid drug name."
)


# =========================================================
# DRUG COMPARISON INPUT
# =========================================================

st.sidebar.divider()

st.sidebar.subheader("🔬 Drug Comparison")

compare_enabled = st.sidebar.checkbox(
    "Enable Drug Comparison"
)

comparison_drug = ""
comparison_smiles = ""

if compare_enabled:

    comparison_drug = st.sidebar.text_input(
        "Enter Second Drug Name:",
        ""
    )

    if comparison_drug:

        comparison_smiles = get_smiles(
            comparison_drug
        )

        if comparison_smiles:

            st.sidebar.text_input(
                "Second Drug SMILES:",
                value=comparison_smiles,
                disabled=True
            )

        else:

            st.sidebar.error(
                "Second drug not found."
            )


# =========================================================
# MAIN MOLECULE
# =========================================================

mol = None

if smiles_input:

    mol = Chem.MolFromSmiles(
        smiles_input
    )


# =========================================================
# SECOND MOLECULE
# =========================================================

comparison_mol = None

if comparison_smiles:

    comparison_mol = Chem.MolFromSmiles(
        comparison_smiles
    )


# =========================================================
# NO DRUG SELECTED
# =========================================================

if mol is None:

    st.info(
        "👈 Enter a valid drug name in the sidebar "
        "to start the analysis."
    )

else:

    # =====================================================
    # STRUCTURE
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🧬 Molecular Structure")

        if structure_type == "2D":

            image = Draw.MolToImage(
                mol,
                size=(500, 400)
            )

            st.image(image)

        else:

            mol_3d = Chem.AddHs(mol)

            try:

                AllChem.EmbedMolecule(
                    mol_3d,
                    randomSeed=42
                )

                AllChem.MMFFOptimizeMolecule(
                    mol_3d
                )

                mol_block = Chem.MolToMolBlock(
                    mol_3d
                )

                view = py3Dmol.view(
                    width=500,
                    height=500
                )

                view.addModel(
                    mol_block,
                    "mol"
                )

                view.setStyle({
                    "stick": {},
                    "sphere": {
                        "scale": 0.3
                    }
                })

                view.zoomTo()

                stmol.showmol(
                    view,
                    height=500,
                    width=500
                )

            except Exception as e:

                st.error(
                    f"3D structure generation error: {e}"
                )


    # =====================================================
    # RIGHT SIDE
    # SECOND DRUG IN COMPARISON / PROPERTIES IN SINGLE MODE
    # =====================================================

    with col2:

        if compare_enabled and comparison_mol is not None:

            st.subheader(
                "🧬 Second Drug Structure"
            )

            if structure_type == "2D":

                comparison_image = Draw.MolToImage(
                    comparison_mol,
                    size=(500, 400)
                )

                st.image(comparison_image)

            else:

                comparison_mol_3d = Chem.AddHs(
                    comparison_mol
                )

                try:

                    AllChem.EmbedMolecule(
                        comparison_mol_3d,
                        randomSeed=42
                    )

                    AllChem.MMFFOptimizeMolecule(
                        comparison_mol_3d
                    )

                    comparison_mol_block = (
                        Chem.MolToMolBlock(
                            comparison_mol_3d
                        )
                    )

                    comparison_view = py3Dmol.view(
                        width=500,
                        height=500
                    )

                    comparison_view.addModel(
                        comparison_mol_block,
                        "mol"
                    )

                    comparison_view.setStyle({
                        "stick": {},
                        "sphere": {
                            "scale": 0.3
                        }
                    })

                    comparison_view.zoomTo()

                    stmol.showmol(
                        comparison_view,
                        height=500,
                        width=500
                    )

                except Exception as e:

                    st.error(
                        f"Second drug 3D structure generation error: {e}"
                    )

        


    # =====================================================
    # MAIN MOLECULAR PROPERTIES
    # ONLY WHEN COMPARISON IS OFF
    # =====================================================

    if not compare_enabled:

        with col2:

            st.subheader("📊 Molecular Properties")

            mw = Descriptors.MolWt(mol)

            molecular_formula = (
                rdMolDescriptors.CalcMolFormula(mol)
            )

            logp = Descriptors.MolLogP(mol)

            hdonors = Descriptors.NumHDonors(mol)

            hacceptors = Descriptors.NumHAcceptors(mol)

            tpsa = Descriptors.TPSA(mol)

            df = pd.DataFrame({

                "Property": [
                    "Molecular Formula",
                    "Molecular Weight",
                    "LogP (Lipophilicity)",
                    "H-Bond Donors",
                    "H-Bond Acceptors",
                    "TPSA"
                ],

                "Value": [
                    molecular_formula,
                    round(mw, 2),
                    round(logp, 2),
                    hdonors,
                    hacceptors,
                    round(tpsa, 2)
                ]

            })

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )


        st.divider()


        # =================================================
        # LIPINSKI
        # =================================================

        st.subheader(
            "💊 Lipinski Rule of 5 Assessment"
        )

        violations = 0

        if mw > 500:
            violations += 1

        if logp > 5:
            violations += 1

        if hdonors > 5:
            violations += 1

        if hacceptors > 10:
            violations += 1


        if violations == 0:

            st.success(
                "✅ No Lipinski Rule of 5 violations detected."
            )

        else:

            st.warning(
                f"⚠️ {violations} Lipinski Rule of 5 "
                "violation(s) detected."
            )


        st.info(
            "Lipinski's Rule of 5 is a drug-likeness heuristic "
            "and does not prove that a compound will become "
            "a successful drug."
        )

        st.divider()


        # =================================================
        # AI DRUG-LIKENESS
        # =================================================

        st.subheader(
            "🤖 AI Drug-Likeness Prediction"
        )

        prediction = None
        confidence = None

        if model_loaded:

            input_data = pd.DataFrame({

                "MW": [mw],
                "LogP": [logp],
                "HBD": [hdonors],
                "HBA": [hacceptors],
                "TPSA": [tpsa]

            })

            try:

                prediction = model.predict(
                    input_data
                )[0]

                if prediction == 1:

                    st.success(
                        "🤖 AI Prediction: Drug-like profile"
                    )

                else:

                    st.warning(
                        "🤖 AI Prediction: Less drug-like profile"
                    )


                if hasattr(model, "predict_proba"):

                    probability = model.predict_proba(
                        input_data
                    )[0]

                    confidence = max(
                        probability
                    ) * 100

                    st.metric(
                        "Model Prediction Confidence",
                        f"{confidence:.1f}%"
                    )

            except Exception as e:

                st.error(
                    f"AI prediction error: {e}"
                )

        else:

            st.error(
                "ML model not found."
            )


        st.divider()


        # =================================================
        # SINGLE DRUG ADMET
        # =================================================

        st.subheader(
            "🧪 ADMET-AI Analysis"
        )

        if admet_loaded:

            try:

                admet_results = admet_model.predict(
                    smiles=smiles_input
                )

                if hasattr(
                    admet_results,
                    "to_dict"
                ):

                    admet_results = (
                        admet_results.to_dict()
                    )


                def admet_value(*names):

                    for name in names:

                        if name in admet_results:

                            return admet_results[name]

                    return None


                def show_probability(
                    label,
                    value
                ):

                    if value is None:

                        st.caption(
                            f"{label}: Not available"
                        )

                        return

                    try:

                        value = float(value)

                        st.metric(
                            label,
                            f"{value * 100:.1f}%"
                        )

                        st.progress(
                            min(
                                max(value, 0.0),
                                1.0
                            )
                        )

                    except (
                        TypeError,
                        ValueError
                    ):

                        st.metric(
                            label,
                            str(value)
                        )


                def show_value(
                    label,
                    value,
                    unit=""
                ):

                    if value is None:

                        st.caption(
                            f"{label}: Not available"
                        )

                        return

                    try:

                        number = float(value)

                        st.metric(
                            label,
                            f"{number:.2f} {unit}"
                        )

                    except (
                        TypeError,
                        ValueError
                    ):

                        st.metric(
                            label,
                            str(value)
                        )


                # =========================================
                # ABSORPTION
                # =========================================

                st.markdown("## 🟢 Absorption")

                st.caption(
                    "Predicted gastrointestinal absorption "
                    "and membrane transport."
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    show_probability(
                        "GI / Human Intestinal Absorption",
                        admet_value("HIA_Hou")
                    )

                with c2:

                    show_probability(
                        "Oral Bioavailability",
                        admet_value(
                            "Bioavailability_Ma"
                        )
                    )

                with c3:

                    show_probability(
                        "P-gp Substrate",
                        admet_value(
                            "Pgp_Broccatelli"
                        )
                    )


                c4, c5 = st.columns(2)

                with c4:

                    show_value(
                        "Caco-2 Permeability",
                        admet_value(
                            "Caco2_Wang"
                        ),
                        "log cm/s"
                    )

                with c5:

                    show_probability(
                        "PAMPA Permeability",
                        admet_value(
                            "PAMPA_NCATS"
                        )
                    )


                st.divider()


                # =========================================
                # DISTRIBUTION
                # =========================================

                st.markdown("## 🔵 Distribution")

                c1, c2, c3 = st.columns(3)

                with c1:

                    show_probability(
                        "Blood-Brain Barrier (BBB)",
                        admet_value(
                            "BBB_Martins"
                        )
                    )

                with c2:

                    show_value(
                        "Volume of Distribution",
                        admet_value(
                            "VDss_Lombardo"
                        ),
                        "L/kg"
                    )

                with c3:

                    show_value(
                        "Plasma Protein Binding",
                        admet_value(
                            "PPBR_AZ"
                        ),
                        "%"
                    )


                st.divider()


                # =========================================
                # METABOLISM
                # =========================================

                st.markdown("## 🟣 Metabolism")

                c1, c2, c3, c4, c5 = st.columns(5)

                cyp_data = [
                    ("CYP1A2", "CYP1A2_Veith"),
                    ("CYP2C9", "CYP2C9_Veith"),
                    ("CYP2C19", "CYP2C19_Veith"),
                    ("CYP2D6", "CYP2D6_Veith"),
                    ("CYP3A4", "CYP3A4_Veith")
                ]

                for column, (
                    label,
                    key
                ) in zip(
                    [c1, c2, c3, c4, c5],
                    cyp_data
                ):

                    with column:

                        show_probability(
                            f"{label} Inhibition",
                            admet_value(key)
                        )


                st.divider()


                # =========================================
                # EXCRETION
                # =========================================

                st.markdown("## 🟠 Excretion")

                c1, c2, c3 = st.columns(3)

                with c1:

                    show_value(
    "Estimated Half-Life",
   abs(float(admet_value("Half_Life_Obach"))),
    
    "hours"
)

                with c2:

                    show_value(
                        "Hepatocyte Clearance",
                        admet_value(
                            "Clearance_Hepatocyte_AZ"
                        ),
                        "µL/min/10⁶ cells"
                    )

                with c3:

                    show_value(
                        "Microsomal Clearance",
                        admet_value(
                            "Clearance_Microsome_AZ"
                        ),
                        "mL/min/g"
                    )


                st.divider()


                # =========================================
                # TOXICITY
                # =========================================

                st.markdown(
                    "## 🔴 Toxicity & Safety"
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    show_probability(
                        "hERG Risk",
                        admet_value("hERG")
                    )

                with c2:

                    show_probability(
                        "AMES Mutagenicity",
                        admet_value("AMES")
                    )

                with c3:

                    show_probability(
                        "Drug-Induced Liver Injury",
                        admet_value("DILI")
                    )


                c4, c5, c6 = st.columns(3)

                with c4:

                    show_probability(
                        "Clinical Toxicity",
                        admet_value("ClinTox")
                    )

                with c5:

                    show_probability(
                        "Skin Reaction",
                        admet_value(
                            "Skin_Reaction"
                        )
                    )

                with c6:

                    show_probability(
                        "Carcinogenicity",
                        admet_value(
                            "Carcinogens_Lagunin"
                        )
                    )


                st.divider()

                st.success(
                    f"✅ ADMET-AI analysis completed for "
                    f"{drug_name}."
                )

                st.info(
                    "⚠️ ADMET-AI predictions are computational "
                    "estimates for educational/research purposes "
                    "and should not be interpreted as clinical "
                    "or therapeutic measurements."
                )


            except Exception as e:

                st.error(
                    f"❌ ADMET-AI prediction error: {e}"
                )

        else:

            st.error(
                "ADMET-AI model could not be loaded."
            )


    # =====================================================
    # COMPARISON MODE
    # =====================================================

    if compare_enabled:

        # =================================================
        # CHECK SECOND DRUG
        # =================================================

        if comparison_mol is None:

            st.warning(
                "👈 Enter a valid second drug name "
                "to start comparison."
            )

        else:

            # =============================================
            # SECOND DRUG PROPERTIES
            # =============================================

            comparison_mw = Descriptors.MolWt(
                comparison_mol
            )

            comparison_formula = (
                rdMolDescriptors.CalcMolFormula(
                    comparison_mol
                )
            )

            comparison_logp = Descriptors.MolLogP(
                comparison_mol
            )

            comparison_hdonors = (
                Descriptors.NumHDonors(
                    comparison_mol
                )
            )

            comparison_hacceptors = (
                Descriptors.NumHAcceptors(
                    comparison_mol
                )
            )

            comparison_tpsa = Descriptors.TPSA(
                comparison_mol
            )


            # =============================================
            # COMPARISON MOLECULAR PROPERTIES
            # =============================================

            st.divider()

            st.subheader(
                "🔬 Drug Comparison - Molecular Properties"
            )

            st.caption(
                f"Comparing {drug_name} with "
                f"{comparison_drug}"
            )

            comparison_df = pd.DataFrame({

                "Property": [
                    "Molecular Formula",
                    "Molecular Weight",
                    "LogP (Lipophilicity)",
                    "H-Bond Donors",
                    "H-Bond Acceptors",
                    "TPSA"
                ],

                drug_name: [
                    rdMolDescriptors.CalcMolFormula(mol),
                    round(
                        Descriptors.MolWt(mol),
                        2
                    ),
                    round(
                        Descriptors.MolLogP(mol),
                        2
                    ),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    round(
                        Descriptors.TPSA(mol),
                        2
                    )
                ],

                comparison_drug: [
                    comparison_formula,
                    round(
                        comparison_mw,
                        2
                    ),
                    round(
                        comparison_logp,
                        2
                    ),
                    comparison_hdonors,
                    comparison_hacceptors,
                    round(
                        comparison_tpsa,
                        2
                    )
                ]

            })

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )


            # =============================================
            # AI & LIPINSKI COMPARISON
            # =============================================

            st.subheader(
                "🤖 AI & Lipinski Comparison"
            )

            # First drug values

            main_mw = Descriptors.MolWt(mol)
            main_logp = Descriptors.MolLogP(mol)
            main_hdonors = Descriptors.NumHDonors(mol)
            main_hacceptors = Descriptors.NumHAcceptors(mol)
            main_tpsa = Descriptors.TPSA(mol)

            main_violations = 0

            if main_mw > 500:
                main_violations += 1

            if main_logp > 5:
                main_violations += 1

            if main_hdonors > 5:
                main_violations += 1

            if main_hacceptors > 10:
                main_violations += 1


            # Second drug Lipinski

            comparison_violations = 0

            if comparison_mw > 500:
                comparison_violations += 1

            if comparison_logp > 5:
                comparison_violations += 1

            if comparison_hdonors > 5:
                comparison_violations += 1

            if comparison_hacceptors > 10:
                comparison_violations += 1


            # AI prediction

            main_prediction_text = "Not Available"
            comparison_prediction_text = "Not Available"

            main_confidence = None
            comparison_confidence = None


            if model_loaded:

                main_input = pd.DataFrame({

                    "MW": [main_mw],
                    "LogP": [main_logp],
                    "HBD": [main_hdonors],
                    "HBA": [main_hacceptors],
                    "TPSA": [main_tpsa]

                })

                comparison_input = pd.DataFrame({

                    "MW": [comparison_mw],
                    "LogP": [comparison_logp],
                    "HBD": [comparison_hdonors],
                    "HBA": [comparison_hacceptors],
                    "TPSA": [comparison_tpsa]

                })


                try:

                    main_prediction = model.predict(
                        main_input
                    )[0]

                    comparison_prediction = model.predict(
                        comparison_input
                    )[0]


                    if main_prediction == 1:

                        main_prediction_text = (
                            "Drug-like profile"
                        )

                    else:

                        main_prediction_text = (
                            "Less drug-like profile"
                        )


                    if comparison_prediction == 1:

                        comparison_prediction_text = (
                            "Drug-like profile"
                        )

                    else:

                        comparison_prediction_text = (
                            "Less drug-like profile"
                        )


                    if hasattr(
                        model,
                        "predict_proba"
                    ):

                        main_probability = (
                            model.predict_proba(
                                main_input
                            )[0]
                        )

                        comparison_probability = (
                            model.predict_proba(
                                comparison_input
                            )[0]
                        )

                        main_confidence = (
                            max(main_probability) * 100
                        )

                        comparison_confidence = (
                            max(comparison_probability) * 100
                        )

                except Exception as e:

                    st.error(
                        f"Comparison AI error: {e}"
                    )


            ai_lipinski_df = pd.DataFrame({

                "Assessment": [
                    "Lipinski Violations",
                    "AI Drug-Likeness",
                    "AI Confidence"
                ],

                drug_name: [
                    main_violations,
                    main_prediction_text,
                    (
                        f"{main_confidence:.1f}%"
                        if main_confidence is not None
                        else "Not Available"
                    )
                ],

                comparison_drug: [
                    comparison_violations,
                    comparison_prediction_text,
                    (
                        f"{comparison_confidence:.1f}%"
                        if comparison_confidence is not None
                        else "Not Available"
                    )
                ]

            })

            st.dataframe(
                ai_lipinski_df,
                use_container_width=True,
                hide_index=True
            )


            # =============================================
            # SECOND DRUG ADMET
            # =============================================

            comparison_admet_results = None

            if admet_loaded:

                try:

                    comparison_admet_results = (
                        admet_model.predict(
                            smiles=comparison_smiles
                        )
                    )

                    if hasattr(
                        comparison_admet_results,
                        "to_dict"
                    ):

                        comparison_admet_results = (
                            comparison_admet_results.to_dict()
                        )

                except Exception as e:

                    st.warning(
                        "Second drug ADMET prediction "
                        f"unavailable: {e}"
                    )


            # =============================================
            # ADMET COMPARISON
            # =============================================

            if comparison_admet_results is not None:

                st.divider()

                st.subheader(
                    "🧪 ADMET Comparison"
                )

                st.caption(
                    f"Comparing ADMET-AI predictions: "
                    f"{drug_name} vs {comparison_drug}"
                )


                # Main drug ADMET

                try:

                    main_admet_results = (
                        admet_model.predict(
                            smiles=smiles_input
                        )
                    )

                    if hasattr(
                        main_admet_results,
                        "to_dict"
                    ):

                        main_admet_results = (
                            main_admet_results.to_dict()
                        )

                except Exception:

                    main_admet_results = {}


                def main_admet_value(*names):

                    for name in names:

                        if name in main_admet_results:

                            return main_admet_results[name]

                    return None


                def comparison_admet_value(*names):

                    for name in names:

                        if name in comparison_admet_results:

                            return comparison_admet_results[name]

                    return None


                def show_comparison_probability(
                    label,
                    main_value,
                    second_value
                ):

                    c1, c2 = st.columns(2)

                    with c1:

                        st.markdown(
                            f"*{drug_name}*"
                        )

                        if main_value is None:

                            st.caption(
                                f"{label}: Not available"
                            )

                        else:

                            try:

                                value = float(
                                    main_value
                                )

                                st.metric(
                                    label,
                                    f"{value * 100:.1f}%"
                                )

                                st.progress(
                                    min(
                                        max(
                                            value,
                                            0.0
                                        ),
                                        1.0
                                    )
                                )

                            except (
                                TypeError,
                                ValueError
                            ):

                                st.metric(
                                    label,
                                    str(main_value)
                                )


                    with c2:

                        st.markdown(
                            f"*{comparison_drug}*"
                        )

                        if second_value is None:

                            st.caption(
                                f"{label}: Not available"
                            )

                        else:

                            try:

                                value = float(
                                    second_value
                                )

                                st.metric(
                                    label,
                                    f"{value * 100:.1f}%"
                                )

                                st.progress(
                                    min(
                                        max(
                                            value,
                                            0.0
                                        ),
                                        1.0
                                    )
                                )

                            except (
                                TypeError,
                                ValueError
                            ):

                                st.metric(
                                    label,
                                    str(second_value)
                                )


                def show_comparison_value(
                    label,
                    main_value,
                    second_value,
                    unit=""
                ):

                    c1, c2 = st.columns(2)

                    with c1:

                        st.markdown(
                            f"*{drug_name}*"
                        )

                        if main_value is None:

                            st.caption(
                                f"{label}: Not available"
                            )

                        else:

                            try:

                                value = float(
                                    main_value
                                )

                                st.metric(
                                    label,
                                    f"{value:.2f} {unit}"
                                )

                            except (
                                TypeError,
                                ValueError
                            ):

                                st.metric(
                                    label,
                                    str(main_value)
                                )


                    with c2:

                        st.markdown(
                            f"*{comparison_drug}*"
                        )

                        if second_value is None:

                            st.caption(
                                f"{label}: Not available"
                            )

                        else:

                            try:

                                value = float(
                                    second_value
                                )

                                st.metric(
                                    label,
                                    f"{value:.2f} {unit}"
                                )

                            except (
                                TypeError,
                                ValueError
                            ):

                                st.metric(
                                    label,
                                    str(second_value)
                                )


                # =========================================
                # ABSORPTION
                # =========================================

                with st.expander(
                    "🟢 Absorption",
                    expanded=True
                ):

                    show_comparison_probability(
                        "GI / Human Intestinal Absorption",
                        main_admet_value(
                            "HIA_Hou"
                        ),
                        comparison_admet_value(
                            "HIA_Hou"
                        )
                    )

                    show_comparison_probability(
                        "Oral Bioavailability",
                        main_admet_value(
                            "Bioavailability_Ma"
                        ),
                        comparison_admet_value(
                            "Bioavailability_Ma"
                        )
                    )

                    show_comparison_probability(
                        "P-gp Substrate",
                        main_admet_value(
                            "Pgp_Broccatelli"
                        ),
                        comparison_admet_value(
                            "Pgp_Broccatelli"
                        )
                    )

                    show_comparison_value(
                        "Caco-2 Permeability",
                        main_admet_value(
                            "Caco2_Wang"
                        ),
                        comparison_admet_value(
                            "Caco2_Wang"
                        ),
                        "log cm/s"
                    )

                    show_comparison_probability(
                        "PAMPA Permeability",
                        main_admet_value(
                            "PAMPA_NCATS"
                        ),
                        comparison_admet_value(
                            "PAMPA_NCATS"
                        )
                    )


                # =========================================
                # DISTRIBUTION
                # =========================================

                with st.expander(
                    "🔵 Distribution",
                    expanded=False
                ):

                    show_comparison_probability(
                        "Blood-Brain Barrier (BBB)",
                        main_admet_value(
                            "BBB_Martins"
                        ),
                        comparison_admet_value(
                            "BBB_Martins"
                        )
                    )

                    show_comparison_value(
                        "Volume of Distribution",
                        main_admet_value(
                            "VDss_Lombardo"
                        ),
                        comparison_admet_value(
                            "VDss_Lombardo"
                        ),
                        "L/kg"
                    )

                    show_comparison_value(
                        "Plasma Protein Binding",
                        main_admet_value(
                            "PPBR_AZ"
                        ),
                        comparison_admet_value(
                            "PPBR_AZ"
                        ),
                        "%"
                    )


                # =========================================
                # METABOLISM
                # =========================================

                with st.expander(
                    "🟣 Metabolism",
                    expanded=False
                ):

                    cyp_data = [
                        (
                            "CYP1A2 Inhibition",
                            "CYP1A2_Veith"
                        ),
                        (
                            "CYP2C9 Inhibition",
                            "CYP2C9_Veith"
                        ),
                        (
                            "CYP2C19 Inhibition",
                            "CYP2C19_Veith"
                        ),
                        (
                            "CYP2D6 Inhibition",
                            "CYP2D6_Veith"
                        ),
                        (
                            "CYP3A4 Inhibition",
                            "CYP3A4_Veith"
                        )
                    ]

                    for label, key in cyp_data:

                        show_comparison_probability(
                            label,
                            main_admet_value(key),
                            comparison_admet_value(key)
                        )


                # =========================================
                # EXCRETION
                # =========================================

                with st.expander(
                    "🟠 Excretion",
                    expanded=False
                ):

                    show_comparison_value(
    "Estimated Half-Life",
    abs(float(main_admet_value("Half_Life_Obach"))),
    abs(float(comparison_admet_value("Half_Life_Obach"))),
    "hours"
)

                    show_comparison_value(
                        "Hepatocyte Clearance",
                        main_admet_value(
                            "Clearance_Hepatocyte_AZ"
                        ),
                        comparison_admet_value(
                            "Clearance_Hepatocyte_AZ"
                        ),
                        "µL/min/10⁶ cells"
                    )

                    show_comparison_value(
                        "Microsomal Clearance",
                        main_admet_value(
                            "Clearance_Microsome_AZ"
                        ),
                        comparison_admet_value(
                            "Clearance_Microsome_AZ"
                        ),
                        "mL/min/g"
                    )


                # =========================================
                # TOXICITY
                # =========================================

                with st.expander(
                    "🔴 Toxicity & Safety",
                    expanded=False
                ):

                    show_comparison_probability(
                        "hERG Risk",
                        main_admet_value("hERG"),
                        comparison_admet_value("hERG")
                    )

                    show_comparison_probability(
                        "AMES Mutagenicity",
                        main_admet_value("AMES"),
                        comparison_admet_value("AMES")
                    )

                    show_comparison_probability(
                        "Drug-Induced Liver Injury",
                        main_admet_value("DILI"),
                        comparison_admet_value("DILI")
                    )

                    show_comparison_probability(
                        "Clinical Toxicity",
                        main_admet_value("ClinTox"),
                        comparison_admet_value("ClinTox")
                    )

                    show_comparison_probability(
                        "Skin Reaction",
                        main_admet_value(
                            "Skin_Reaction"
                        ),
                        comparison_admet_value(
                            "Skin_Reaction"
                        )
                    )

                    show_comparison_probability(
                        "Carcinogenicity",
                        main_admet_value(
                            "Carcinogens_Lagunin"
                        ),
                        comparison_admet_value(
                            "Carcinogens_Lagunin"
                        )
                    )


                st.success(
                    "✅ ADMET comparison completed."
                )


    # =====================================================
    # PROJECT SUMMARY
    # =====================================================

    st.divider()

    st.subheader(
        "📋 Project Summary"
    )

    if compare_enabled:

        summary = pd.DataFrame({

            "Analysis": [
                "Drug Comparison",
                "AI & Lipinski Comparison",
                "ADMET Comparison"
            ],

            "Status": [
                "Completed",
                "Completed",
                "Completed"
            ]

        })

    else:

        summary = pd.DataFrame({

            "Analysis": [
                "Molecular Structure",
                "Molecular Properties",
                "Lipinski Assessment",
                "ML Drug-Likeness",
                "ADMET Indicators"
            ],

            "Status": [
                "Completed",
                "Completed",
                "Completed",
                "Completed"
                if model_loaded
                else "Model Missing",
                "Educational Analysis"
            ]

        })


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.warning(
        "⚠️ Project prototype only: The ML and ADMET-related "
        "outputs are for educational/research demonstration "
        "and must not be used for clinical, diagnostic, "
        "therapeutic or drug-development decisions."
    )


    # =====================================================
    # FOOTER
    # =====================================================

    


    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    st.subheader(
        "📥 Download Analysis Report"
    )

    if compare_enabled:

        report_data = pd.DataFrame({

            "Parameter": [
                "First Drug",
                "Second Drug",
                "Comparison Mode"
            ],

            "Result": [
                drug_name,
                comparison_drug,
                "Enabled"
            ]

        })

        file_name = "drug_comparison_analysis.csv"

    else:

        report_data = pd.DataFrame({

            "Parameter": [
                "SMILES",
                "Molecular Weight",
                "LogP",
                "H-Bond Donors",
                "H-Bond Acceptors",
                "TPSA"
            ],

            "Result": [
                smiles_input,
                round(
                    Descriptors.MolWt(mol),
                    2
                ),
                round(
                    Descriptors.MolLogP(mol),
                    2
                ),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                round(
                    Descriptors.TPSA(mol),
                    2
                )
            ]

        })

        file_name = "drug_discovery_analysis.csv"


    csv_data = report_data.to_csv(
        index=False
    )


    st.download_button(
        "📥 Download CSV Report",
        data=csv_data,
        file_name=file_name,
        mime="text/csv"
    )
