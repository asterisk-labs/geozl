# Frame compatibility

Starting with 0.13.0, every GeoZL release will read frames written by earlier
releases from 0.13.0 onward.

Released CTids are never reassigned. An incompatible codec layout receives a
new CTid. Encoders may change their output, and older readers are not expected
to understand codecs introduced later.

This guarantee covers GeoZL codec data. OpenZL container compatibility follows
OpenZL. The C source API has a separate [stability policy](c-api.md).
