from typing import List, Dict
from query_builder.elasticsearch.generate_script import generate_script
from utils.scaler import get_z_value
from config.config import Config, TrialsQueryConfig, MarcoQueryConfig, apply_config


class Query:
    topics: Dict[int, Dict[str, str]]
    query_type: str
    fields: List[int]
    query_funcs: Dict
    config: Config

    def generate_query(self, *args, **kwargs):
        raise NotImplementedError

    def get_query_type(self, *args, **kwargs):
        return self.query_funcs[self.query_type](*args, **kwargs)


class TrialsQuery(Query):
    mappings: List[str]
    config: TrialsQueryConfig

    def __init__(self, topics, query_type, config=None, *args, **kwargs):
        self.query_type = query_type
        self.config = config
        self.topics = topics
        self.fields = []
        self.mappings = [
            "HasExpandedAccess",
            "BriefSummary.Textblock",
            "CompletionDate.Type",
            "OversightInfo.Text",
            "OverallContactBackup.PhoneExt",
            "RemovedCountries.Text",
            "SecondaryOutcome",
            "Sponsors.LeadSponsor.Text",
            "BriefTitle",
            "IDInfo.NctID",
            "IDInfo.SecondaryID",
            "OverallContactBackup.Phone",
            "Eligibility.StudyPop.Textblock",
            "DetailedDescription.Textblock",
            "Eligibility.MinimumAge",
            "Sponsors.Collaborator",
            "Reference",
            "Eligibility.Criteria.Textblock",
            "XMLName.Space",
            "Rank",
            "OverallStatus",
            "InterventionBrowse.Text",
            "Eligibility.Text",
            "Intervention",
            "BiospecDescr.Textblock",
            "ResponsibleParty.NameTitle",
            "NumberOfArms",
            "ResponsibleParty.ResponsiblePartyType",
            "IsSection801",
            "Acronym",
            "Eligibility.MaximumAge",
            "DetailedDescription.Text",
            "StudyDesign",
            "OtherOutcome",
            "VerificationDate",
            "ConditionBrowse.MeshTerm",
            "Enrollment.Text",
            "IDInfo.Text",
            "ConditionBrowse.Text",
            "FirstreceivedDate",
            "NumberOfGroups",
            "OversightInfo.HasDmc",
            "PrimaryCompletionDate.Text",
            "ResultsReference",
            "Eligibility.StudyPop.Text",
            "IsFdaRegulated",
            "WhyStopped",
            "ArmGroup",
            "OverallContact.LastName",
            "Phase",
            "RemovedCountries.Country",
            "InterventionBrowse.MeshTerm",
            "Eligibility.HealthyVolunteers",
            "Location",
            "OfficialTitle",
            "OverallContact.Email",
            "RequiredHeader.Text",
            "RequiredHeader.URL",
            "LocationCountries.Country",
            "OverallContact.PhoneExt",
            "Condition",
            "PrimaryOutcome",
            "LocationCountries.Text",
            "BiospecDescr.Text",
            "IDInfo.OrgStudyID",
            "Link",
            "OverallContact.Phone",
            "Source",
            "ResponsibleParty.InvestigatorAffiliation",
            "StudyType",
            "FirstreceivedResultsDate",
            "Enrollment.Type",
            "Eligibility.Gender",
            "OverallContactBackup.LastName",
            "Keyword",
            "BiospecRetention",
            "CompletionDate.Text",
            "OverallContact.Text",
            "RequiredHeader.DownloadDate",
            "Sponsors.Text",
            "Text",
            "Eligibility.SamplingMethod",
            "LastchangedDate",
            "ResponsibleParty.InvestigatorFullName",
            "StartDate",
            "RequiredHeader.LinkText",
            "OverallOfficial",
            "Sponsors.LeadSponsor.AgencyClass",
            "OverallContactBackup.Text",
            "Eligibility.Criteria.Text",
            "XMLName.Local",
            "OversightInfo.Authority",
            "PrimaryCompletionDate.Type",
            "ResponsibleParty.Organization",
            "IDInfo.NctAlias",
            "ResponsibleParty.Text",
            "TargetDuration",
            "Sponsors.LeadSponsor.Agency",
            "BriefSummary.Text",
            "OverallContactBackup.Email",
            "ResponsibleParty.InvestigatorTitle",
        ]

        self.best_recall_fields = [
            "LocationCountries.Country",
            "BiospecRetention",
            "DetailedDescription.Textblock",
            "HasExpandedAccess",
            "ConditionBrowse.MeshTerm",
            "RequiredHeader.LinkText",
            "WhyStopped",
            "BriefSummary.Textblock",
            "Eligibility.Criteria.Textblock",
            "OfficialTitle",
            "Eligibility.MaximumAge",
            "Eligibility.StudyPop.Textblock",
            "BiospecDescr.Textblock",
            "BriefTitle",
            "Eligibility.MinimumAge",
            "ResponsibleParty.Organization",
            "TargetDuration",
            "Condition",
            "IDInfo.OrgStudyID",
            "Keyword",
            "Source",
            "Sponsors.LeadSponsor.Agency",
            "ResponsibleParty.InvestigatorAffiliation",
            "OversightInfo.Authority",
            "OversightInfo.HasDmc",
            "OverallContact.Phone",
            "Phase",
            "OverallContactBackup.LastName",
            "Acronym",
            "InterventionBrowse.MeshTerm",
            "RemovedCountries.Country",
        ]
        self.best_map_fields = [
            "Eligibility.Gender",
            "LocationCountries.Country",
            "DetailedDescription.Textblock",
            "BriefSummary.Textblock",
            "ConditionBrowse.MeshTerm",
            "Eligibility.Criteria.Textblock",
            "InterventionBrowse.MeshTerm",
            "StudyType",
            "IsFdaRegulated",
            "HasExpandedAccess",
            "RequiredHeader.LinkText",
            "BiospecRetention",
            "OfficialTitle",
            "Eligibility.SamplingMethod",
            "Eligibility.StudyPop.Textblock",
            "Condition",
            "Eligibility.MinimumAge",
            "Keyword",
            "Eligibility.MaximumAge",
            "BriefTitle",
        ]
        self.best_embed_fields = [
            "WhyStopped",
            "HasExpandedAccess",
            "BiospecRetention",
            "BriefSummary.Textblock",
            "LocationCountries.Country",
            "ConditionBrowse.MeshTerm",
            "DetailedDescription.Textblock",
            "RequiredHeader.LinkText",
            "Eligibility.Criteria.Textblock",
        ]

        self.sensible = [
            "BriefSummary.Textblock" "BriefTitle",
            "Eligibility.StudyPop.Textblock",
            "DetailedDescription.Textblock",
            "Eligibility.MinimumAge",
            "Eligibility.Criteria.Textblock",
            "InterventionBrowse.Text",
            "Eligibility.Text",
            "BiospecDescr.Textblock",
            "Eligibility.MaximumAge",
            "DetailedDescription.Text",
            "ConditionBrowse.MeshTerm",
            "ConditionBrowse.Text",
            "Eligibility.StudyPop.Text",
            "InterventionBrowse.MeshTerm",
            "OfficialTitle",
            "Condition",
            "PrimaryOutcome",
            "BiospecDescr.Text",
            "Eligibility.Gender",
            "Keyword",
            "BiospecRetention",
            "Eligibility.Criteria.Text",
            "BriefSummary.Text",
        ]

        self.sensible_embed = [
            "BriefSummary.Textblock" "BriefTitle",
            "Eligibility.StudyPop.Textblock",
            "DetailedDescription.Textblock",
            "Eligibility.Criteria.Textblock",
            "InterventionBrowse.Text",
            "Eligibility.Text",
            "BiospecDescr.Textblock",
            "DetailedDescription.Text",
            "ConditionBrowse.MeshTerm",
            "ConditionBrowse.Text",
            "Eligibility.StudyPop.Text",
            "InterventionBrowse.MeshTerm",
            "OfficialTitle",
            "Condition",
            "PrimaryOutcome",
            "BiospecDescr.Text",
            "Keyword",
            "BiospecRetention",
            "Eligibility.Criteria.Text",
            "BriefSummary.Text",
        ]

        self.sensible_embed_safe = list(
            set(self.best_recall_fields).intersection(set(self.sensible_embed))
        )

        self.query_funcs = {
            "query": self.generate_query,
            "ablation": self.generate_query_ablation,
            "embedding": self.generate_query_embedding,
        }

        print(self.sensible_embed_safe)

        self.field_usage = {
            "best_recall_fields": self.best_recall_fields,
            "all": self.mappings,
            "best_map_fields": self.best_map_fields,
            "best_embed_fields": self.best_embed_fields,
            "sensible": self.sensible,
            "sensible_embed": self.sensible_embed,
            "sensible_embed_safe": self.sensible_embed_safe,
        }

    @apply_config
    def generate_query(self, topic_num, query_field_usage, **kwargs):
        fields = self.field_usage[query_field_usage]

        should = {"should": []}

        qfield = list(self.topics[topic_num].keys())[0]
        query = self.topics[topic_num][qfield]

        for i, field in enumerate(fields):
            should["should"].append(
                {
                    "match": {
                        f"{field}": {
                            "query": query,
                        }
                    }
                }
            )

        query = {
            "query": {
                "bool": should,
            }
        }

        return query

    def generate_query_ablation(self, topic_num, **kwargs):
        query = {"query": {"match": {}}}

        for field in self.fields:
            query["query"]["match"][self.mappings[field]] = ""

        for qfield in self.fields:
            qfield = self.mappings[qfield]
            for field in self.topics[topic_num]:
                query["query"]["match"][qfield] += self.topics[topic_num][field]

        return query

    @apply_config
    def generate_query_embedding(
        self,
        topic_num,
        encoder,
        query_field_usage,
        embed_field_usage,
        cosine_weights: List[float] = None,
        query_weight: List[float] = None,
        norm_weight=2.15,
        ablations=False,
        automatic_scores=None,
        **kwargs,
    ):
        should = {"should": []}

        assert norm_weight or automatic_scores

        query_fields = self.field_usage[query_field_usage]
        embed_fields = self.field_usage[embed_field_usage]

        qfield = list(self.topics[topic_num].keys())[0]
        query = self.topics[topic_num][qfield]

        for i, field in enumerate(query_fields):
            should["should"].append(
                {
                    "match": {
                        f"{field}": {
                            "query": query,
                            "boost": query_weight[i] if query_weight else 1,
                        }
                    }
                }
            )

        if automatic_scores is not None:
            norm_weight = get_z_value(
                cosine_ceiling=len(embed_fields),
                bm25_ceiling=automatic_scores[topic_num],
            )

        params = {
            "weights": cosine_weights if cosine_weights else [1] * len(embed_fields),
            "q_eb": encoder.encode(self.topics[topic_num][qfield]),
            "offset": 1.0,
            "norm_weight": norm_weight,
            "disable_bm25": ablations,
        }

        query = {
            "query": {
                "script_score": {
                    "query": {
                        "bool": should,
                    },
                    "script": generate_script(self.best_embed_fields, params=params),
                },
            }
        }

        return query


class GenericQuery:
    topics: Dict[int, Dict[str, str]]
    config: MarcoQueryConfig

    def __init__(self, topics, config, *args, **kwargs):
        self.mappings = ["Text"]

        self.topics = topics
        self.config = config
        self.query_type = self.config.query_type

        self.embed_mappings = ["Text_Embedding"]

        self.query_funcs = {
            "query": self.generate_query,
            "embedding": self.generate_query_embedding,
        }

    def _generate_base_query(self, topic_num):
        qfield = list(self.topics[topic_num].keys())[0]
        query = self.topics[topic_num][qfield]
        should = {"should": []}

        for i, field in enumerate(self.mappings):
            should["should"].append(
                {
                    "match": {
                        f"{field}": {
                            "query": query,
                        }
                    }
                }
            )

        return qfield, query, should

    def generate_query(self, topic_num, *args, **kwargs):
        _, _, should = self._generate_base_query(topic_num)

        query = {
            "query": {
                "bool": should,
            }
        }

        return query

    @apply_config
    def generate_query_embedding(
        self, topic_num, encoder, norm_weight=2.15, ablations=False, automatic=None
    ):
        qfield, query, should = self._generate_base_query(topic_num)

        if automatic is not None:
            norm_weight = get_z_value(
                cosine_ceiling=len(self.embed_mappings),
                bm25_ceiling=automatic[topic_num],
            )

        params = {
            "weights": [1] * len(self.embed_mappings),
            "q_eb": encoder.encode(self.topics[topic_num][qfield]),
            "offset": 1.0,
            "norm_weight": norm_weight,
            "disable_bm25": ablations,
        }

        query = {
            "query": {
                "script_score": {
                    "query": {
                        "bool": should,
                    },
                    "script": generate_script(self.embed_mappings, params=params),
                },
            }
        }

        return query


class BioRedditQuery(GenericQuery):
    def __init__(self, topics, config, *args, **kwargs):
        super().__init__(topics, config, *args, **kwargs)
        self.mappings = ["Text"]

        self.topics = topics
        self.config = config
        self.query_type = self.config.query_type

        self.embed_mappings = ["Text_Embedding"]

        self.query_funcs = {
            "query": self.generate_query,
            "embedding": self.generate_query_embedding,
        }


class TrecQuery(GenericQuery):
    def __init__(self, topics, config, *args, **kwargs):
        super().__init__(topics, config, *args, **kwargs)
        self.mappings = ["Text"]

        self.topics = topics
        self.config = config
        self.query_type = self.config.query_type

        self.embed_mappings = ["Text_Embedding"]

        self.query_funcs = {
            "query": self.generate_query,
            "embedding": self.generate_query_embedding,
        }
