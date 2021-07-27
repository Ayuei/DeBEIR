import json
from typing import List, Dict

def unwind_mapping(cache, node, key=None):
    if not isinstance(node, dict) or (not key in node):
        cache.append(key)
        return

    cache.append(key)

    for ckey in node:
        unwind_mapping(cache, node[key], ckey)
    return


def parse_mappings(fp):
    cache = []

    with open(fp) as f:
        mappings = json.loads(f.read())

        for mapping in mappings['test_trials']['mappings']['properties']:
            unwind_mapping(cache, node=mappings['test_trials']['mappings']['properties'], key=mapping)
            cache.append(mapping)

    mappings = list(set(cache))
    mappings.sort()

    return mappings


class Query:
    def generate_query(self):
        raise NotImplementedError


class TestTrialsQuery:
    mappings: List[str]
    fields: List[int]
    topics: Dict[int, Dict[str, str]]

    def __init__(self, topics, mappings_path):
        self.topics = topics
        self.fields = []
        self.mappings = ['HasExpandedAccess', 'BriefSummary.Textblock', 'CompletionDate.Type', 'OversightInfo.Text', 'OverallContactBackup.PhoneExt', 'RemovedCountries.Text', 'SecondaryOutcome', 'Sponsors.LeadSponsor.Text', 'BriefTitle', 'IDInfo.NctID', 'IDInfo.SecondaryID', 'OverallContactBackup.Phone', 'Eligibility.StudyPop.Textblock', 'DetailedDescription.Textblock', 'Eligibility.MinimumAge', 'Sponsors.Collaborator', 'Reference', 'Eligibility.Criteria.Textblock', 'XMLName.Space', 'Rank', 'OverallStatus', 'InterventionBrowse.Text', 'Eligibility.Text', 'Intervention', 'BiospecDescr.Textblock', 'ResponsibleParty.NameTitle', 'NumberOfArms', 'ResponsibleParty.ResponsiblePartyType', 'IsSection801', 'Acronym', 'Eligibility.MaximumAge', 'DetailedDescription.Text', 'StudyDesign', 'OtherOutcome', 'VerificationDate', 'ConditionBrowse.MeshTerm', 'Enrollment.Text', 'IDInfo.Text', 'ConditionBrowse.Text', 'FirstreceivedDate', 'NumberOfGroups', 'OversightInfo.HasDmc', 'PrimaryCompletionDate.Text', 'ResultsReference', 'Eligibility.StudyPop.Text', 'IsFdaRegulated', 'WhyStopped', 'ArmGroup', 'OverallContact.LastName', 'Phase', 'RemovedCountries.Country', 'InterventionBrowse.MeshTerm', 'Eligibility.HealthyVolunteers', 'Location', 'OfficialTitle', 'OverallContact.Email', 'RequiredHeader.Text', 'RequiredHeader.URL', 'LocationCountries.Country', 'OverallContact.PhoneExt', 'Condition', 'PrimaryOutcome', 'LocationCountries.Text', 'BiospecDescr.Text', 'IDInfo.OrgStudyID', 'Link', 'OverallContact.Phone', 'Source', 'ResponsibleParty.InvestigatorAffiliation', 'StudyType', 'FirstreceivedResultsDate', 'Enrollment.Type', 'Eligibility.Gender', 'OverallContactBackup.LastName', 'Keyword', 'BiospecRetention', 'CompletionDate.Text', 'OverallContact.Text', 'RequiredHeader.DownloadDate', 'Sponsors.Text', 'Text', 'Eligibility.SamplingMethod', 'LastchangedDate', 'ResponsibleParty.InvestigatorFullName', 'StartDate', 'RequiredHeader.LinkText', 'OverallOfficial', 'Sponsors.LeadSponsor.AgencyClass', 'OverallContactBackup.Text', 'Eligibility.Criteria.Text', 'XMLName.Local', 'OversightInfo.Authority', 'PrimaryCompletionDate.Type', 'ResponsibleParty.Organization', 'IDInfo.NctAlias', 'ResponsibleParty.Text', 'TargetDuration', 'Sponsors.LeadSponsor.Agency', 'BriefSummary.Text', 'OverallContactBackup.Email', 'ResponsibleParty.InvestigatorTitle']
        self.best_fields = ["LocationCountries.Country", "BiospecRetention", "DetailedDescription.Textblock", "HasExpandedAccess",
                            "ConditionBrowse.MeshTerm", "RequiredHeader.LinkText", "WhyStopped", "BriefSummary.Textblock",
                            "Eligibility.Criteria.Textblock", "OfficialTitle", "Eligibility.MaximumAge", "Eligibility.StudyPop.Textblock",
                            "BiospecDescr.Textblock", "BriefTitle", "Eligibility.MinimumAge", "ResponsibleParty.Organization",
                            "TargetDuration", "Condition", "IDInfo.OrgStudyID", "Keyword", "Source", "Sponsors.LeadSponsor.Agency",
                            "ResponsibleParty.InvestigatorAffiliation", "OversightInfo.Authority", "OversightInfo.HasDmc", "OverallContact.Phone",
                            "Phase", "OverallContactBackup.LastName", "Acronym", "InterventionBrowse.MeshTerm", "RemovedCountries.Country"]

    def generate_query(self, topic_num, best_fields=True):
        query = {
            "query": {
                "simple_query_string": {
                    "query": "",
                    "fields": self.best_fields if best_fields else ["*"],
                }
            }
        }

        for field in self.topics[topic_num]:
            query['query']['simple_query_string']["query"] += "("+self.topics[topic_num][field]+") "

        return query

    def generate_query_ablation(self, topic_num):
        query = {
            "query": {
                "match": {
                }
            }
        }

        for field in self.fields:
            query['query']['match'][self.mappings[field]] = ""

        for qfield in self.fields:
            qfield = self.mappings[qfield]
            for field in self.topics[topic_num]:
                query['query']['match'][qfield] += self.topics[topic_num][field]

        return query

    def generate_query_embedding(self, query, encoder, cosine_weights: List[float]=None,
                                 query_weight=1,
                                 expansion="",
                                 norm_weight=2.15):

        assert len(query_weights) == 12
        assert len(cosine_weights) == 9

        return {
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            # Match on title, abstract and fulltext on all three fields
                            # Weights should be added later
                            "should": [
                                {"match": {"title": {"query": q, "boost": query_weights[0]}}},
                                {"match": {"title": {"query": qstn, "boost": query_weights[1]}}},
                                {"match": {"title": {"query": narr, "boost": query_weights[2]}}},
                                {"match": {"title": {"query": expansion, "boost": query_weights[3]}}},
                                {"match": {"abstract": {"query": q, "boost": query_weights[4]}}},
                                {"match": {"abstract": {"query": qstn, "boost": query_weights[5]}}},
                                {"match": {"abstract": {"query": narr, "boost": query_weights[6]}}},
                                {"match": {"abstract": {"query": expansion, "boost": query_weights[7]}}},
                                {"match": {"fulltext": {"query": q, "boost": query_weights[8]}}},
                                {"match": {"fulltext": {"query": qstn, "boost": query_weights[9]}}},
                                {"match": {"fulltext": {"query": narr, "boost": query_weights[10]}}},
                                {"match": {"fulltext": {"query": expansion, "boost": query_weights[11]}}},
                            ],
                        }
                    },
                }
            }
        }


if __name__ == "__main__":
    print(len(parse_mappings("../assets/mapping.json")))
