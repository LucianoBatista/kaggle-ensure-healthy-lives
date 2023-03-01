# De todos os missings que a gente tem...

# Aprofundando o entendimento das variáveis

A ideia é registrar aqui informações sobre as variáveis para ser possível tomar melhores decisões para melhorar nosso modelo de previsão.

## CO_MUN_NOT ou ID_MUNICIP (unit_municipio_code_ibge)

Código do município pelo IBGE onde está localizada a **unidade** que registrou a notificação.

## CO_REGIONA ou ID_REGIONA (unit_regionais_saude_code_ibge)

Código das regionais de saúde dos municípios de notificação. Logo, precisamos ter a indicação do Município para que seja possível identificar a Regional de Saúde.

Uma região de saúde é espaço geográfico contínuo constituído por agrupamentos de Municípios limítrofes, delimitado a partir de identidades culturais, econômicas e sociais e de redes de comunicação e infraestrutura de transportes compartilhados, com a finalidade de integrar a organização e o planejamento de ações e serviços de saúde.

Pode ser de:

- Atenção primária (I)
- Urgência e emergência (II)
- Atenção Psicossocial (III)
- Atenção ambulatorial especializada e hospitalar (IV)
- Vigilância em saúde

## SG_UF_NOT (unit_uf_code_ibge)

Unidade Federativa onde está localizada a Unidade que realizou a notificação.

## CS_SEXO (patient_gender)

Pode ser:

- 1: masculino
- 2: feminino
- 9: ignorado

## COD_IDADE (não consta no documento)

**Analisar e verificar os dados que temos aqui**

## CS_GESTANT (patient_gest_period)

Idade gestacional do paciente

Aqui vai ser muito interessante analisar algumas coisinhas:

- Temos missings aqui?
- Se `patient_gender` == masculino, e `age` <= 9 => 6 (Não se aplica)
- Se `patient_gender` == feminino, e `age` >= 9 => Não deveríamos observar 6 (não se aplica)

## CS_RACA (patient_color)

Cor ou raça declarada pelo paciente:

- 1: Branca
- 2: Preta
- 3: Amarela
- 4: Parda
- 5: Indígena
- 9: Ignorado

Pode ser interessante para verificar algum ponto de viés no modelo gerado

## CS_ESCOL_N (patient_education)

Escolaridade do paciente.

- `age` < 7, preenchido automaticamente com "não se aplica"
- `age` < 7, não devemos encontrar o "não se aplica"

Temos as opções:

- 0: Sem escolaridade/analfabeto
- 1: Fundamental I
- 2: Fundamental II
- 3: Médio
- 4: Superior
- 5: Não se aplica
- 9: Ignorado

## SG_UF (patient_uf)

O preenchimento do CEP faz com que esse campo seja automaticamente preenchido. Podemos verificar isso!

Temos ao todo 31 códigos, isso também pode ser verificado.

## CS_ZONA (patient_zone_residence)

Zona geográfica de residência do paciente.

- 1: Urbana
- 2: Rural
- 3: Periurbana
- 9: Ignorado

## SURTO_SG (não consta no documento)

**Analisar e verificar os dados que temos aqui**

## NOSOCOMIAL (is_nosocomial)

Caso de SRAG com infecção adquirida após internação.

- 1: Sim
- 2: Não
- 9: Ignorado

Quando temos 1 nesse campo, é permitido digitar _data de início dos sintomas posterior a data de internação_.

## AVE_SUINO (is_ave_suino)

Paciente trabalha ou tem contato direto com aves, suínos, ou outro animal?

- 1: Sim
- 2: Não
- 9: Ignorado

## FEBRE (is_febre)

Paciente apresentou febre?

- 1: Sim
- 2: Não
- 9: Ignorado

## TOSSE (is_tosse)

Paciente apresentou tosse?

- 1: Sim
- 2: Não
- 9: Ignorado

## GARGANTA (is_garganta)

Paciente apresentou dor de garganta?

- 1: Sim
- 2: Não
- 9: Ignorado

## DISPNEIA (is_dispneia)

Paciente apresentou dispneia?

- 1: Sim
- 2: Não
- 9: Ignorado

## DESC_RESP (is_respiration_down)

Paciente apresentou desconforto respiratório?

- 1: Sim
- 2: Não
- 9: Ignorado

## SATURACAO (is_o2_saturation_down)

Paciente apresentou saturação 02 < 95%?

- 1: Sim
- 2: Não
- 9: Ignorado

## DIARREIA (is_diarreia)

Paciente apresentou diarréia?

- 1: Sim
- 2: Não
- 9: Ignorado

## VOMITO (is_vomito)

Paciente apresentou vômito?

- 1: Sim
- 2: Não
- 9: Ignorado

## OUTRO_SIN (is_outro_sinal)

Paciente apresentou outro sintomas?

- 1: Sim
- 2: Não
- 9: Ignorado

## OUTRO_DES (outro_sinal_desc)

Campo habilitado quando é selecionado 1 no `is_outro_sinal`

## PUERPERA (is_puerpera)

Mulher é puérpera ou parturiente (mulher que pariu recentemente - até 45 dias de parto)?

Habilitado somente em casos de mulher (`patient_gender` == 2)

## FATOR_RISC (is_fator_risco)

Paciente apresenta algum fator de risco?

- 1: Sim
- 2: Não
- 9: Ignorado

## CARDIOPATI (is_cardio_down)

Paciente possui doença cardiovascular crônica?

- 1: Sim
- 2: Não
- 9: Ignorado

## HEMATOLOGI (is_hemato_down)

Paciente possui doença Hematológica crônica?

## SIND_DOWN (is_sindrome_down)

Paciente possui Síndrome de Down?

- 1: Sim
- 2: Não
- 9: Ignorado

## HEPATICA (is_hepatica_down)

Paciente possui alguma doença hepática crônica?

- 1: Sim
- 2: Não
- 9: Ignorado

## ASMA (is_asma)

Possui asma?

- 1: Sim
- 2: Não
- 9: Ignorado

## DIABETES (is_diabetes)

Possui diabetes?

- 1: Sim
- 2: Não
- 9: Ignorado

## NEUROLOGIC (is_neuro_down)

Possui doença neurológica?

- 1: Sim
- 2: Não
- 9: Ignorado

## PNEUMOPATI

## IMUNODEPRE

## RENAL

## OBESIDADE

## OUT_MORBI

## MORB_DESC

## VACINA

## MAE_VAC

## M_AMAMENTA

## ANTIVIRAL

## TP_ANTIVIR

## HOSPITAL

## UTI

## SUPORT_VEN

## RAIOX_RES

## RAIOX_OUT

## AMOSTRA

## TP_AMOSTRA

## OUT_AMOST

## HISTO_VGM

## PAC_COCBO

## PAC_DSCBO

## OUT_ANIM

## DOR_ABD

## FADIGA

## PERD_OLFT

## PERD_PALA

## TOMO_RES

## TOMO_OUT

## VACINA_COV

## DOSE_1_COV

## DOSE_2_COV

## FNT_IN_COV

## TP_IDADE

## OBES_IMC

## SEM_NOT

## SEM_PRI

## NU_IDADE_N

## CLASSI_FIN
