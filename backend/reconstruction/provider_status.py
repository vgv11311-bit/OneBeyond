class ProviderUnavailableError(ValueError):
    """Retryable external outage, not invalid media or a geometry failure."""
    error_code='analysis_service_unavailable'
    retry_after_seconds=30


class IncompleteReconstructionError(ProviderUnavailableError):
    """Required photo evidence was not validated; no GLB may be published."""
    error_code='analysis_incomplete'


class ProviderQuotaError(RuntimeError):
    """Daily quota cannot be recovered by a short automatic retry."""
    error_code='analysis_quota_exhausted'
    retry_after_seconds=0


def daily_quota_exhausted(error):
    """Inspect structured quota IDs, not the ambiguous HTTP 429 alone."""
    import json
    if getattr(error,'code',None)!=429: return False
    if '_daily_quota' in error.__dict__: return error.__dict__['_daily_quota']
    try:
        payload=json.loads(error.read())
        details=payload.get('error',{}).get('details',[])
        result=any('perday' in str(v.get('quotaId','')).lower()
                   for detail in details for v in detail.get('violations',[]))
    except (ValueError,TypeError,AttributeError,KeyError): result=False
    error._daily_quota=result
    return result


QUOTA_MESSAGE=('Google photo analysis daily quota is exhausted. Your upload and room-width setting are saved. '
               'Wait for the daily quota reset or enable/increase quota in the Google API project, then retry. '
               'No substitute room was generated.')
